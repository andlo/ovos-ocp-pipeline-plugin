import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from urllib.parse import urlsplit, urlunsplit
from dataclasses import dataclass
from pathlib import Path
from os.path import join, dirname, isdir
from threading import RLock
from typing import Tuple, Optional, Dict, List, Union, Any

from ovos_bus_client.apis.ocp import ClassicAudioServiceInterface
from ovos_bus_client.apis.ocp import OCPInterface, OCPQuery
from ovos_bus_client.client import MessageBusClient
from ovos_bus_client.message import Message, dig_for_message
from ovos_bus_client.session import Session, SessionManager
from ovos_config import Configuration
from ovos_plugin_manager.ocp import available_extractors
from ovos_plugin_manager.templates.pipeline import IntentHandlerMatch, ConfidenceMatcherPipeline, PipelinePlugin
from ovos_spec_tools import standardize_lang, closest_lang, voc_match
from ovos_spec_tools.resources import read_resource_file, strip_samples
from ovos_utils.log import LOG, deprecated, log_deprecation
from ovos_utils.fakebus import FakeBus
from ovos_utils.ocp import MediaType, PlaybackType, PlaybackMode, PlayerState, OCP_ID, \
    MediaEntry, Playlist, MediaState, TrackState, dict2entry, PluginStream
from ovos_workshop.app import OVOSAbstractApplication
from ovos_utils.xdg_utils import xdg_data_home
from ovos_config.meta import get_xdg_base
from ahocorasick_ner import AhocorasickNER
from ocp_pipeline.legacy import LegacyCommonPlay
from ocp_pipeline.context_classify import (
    ContextAwareClassifier,
    build_ner_list,
    build_player_status,
)

from ovos_plugin_manager.media_provider import load_media_providers
from ovos_plugin_manager.templates.media_provider import MediaProvider
from ocp_pipeline.bridge import media_type_to_signals, ocp_media_types_for, release_to_ocp_result

# .voc resources shipped with this plugin (locale/<lang>/<name>.voc).
# Matched via ovos-spec-tools voc_match (OVOS-INTENT-2 §4.3 whole-word semantics)
# instead of reaching into ovos-workshop's skill voc_match.
LOCALE_DIR = join(dirname(__file__), "locale")

# cap on how many releases a single MediaProvider can contribute per search,
# applied after the provider returns -- a misbehaving/unbounded provider
# (200-release repro) cannot flood the merged result pool
MAX_PROVIDER_RESULTS = 50

# ceiling for a result from a MediaProvider that never declared the concrete
# media type the user asked for. It is the default `min_score` floor, so such
# a result stays playable when it is the only answer; when some source that
# does claim the type also answered, the undeclared result drops one point
# below the floor instead (see `_demote_undeclared_providers`).
UNDECLARED_PROVIDER_MAX_CONFIDENCE = 50


def _is_initialised(obj) -> bool:
    """True when ``OVOSAbstractApplication.__init__`` ran on ``obj``.

    ``OVOSSkill.__del__`` calls ``shutdown()`` and then ``default_shutdown()``,
    and the garbage collector reaches it on instances whose ``__init__`` never
    finished: this suite skips padatious training with a ``__new__`` bypass, and
    a raise anywhere inside ``__init__`` leaves the same half-built object.
    Both teardown paths then fail on ``intent_service``, and ``__del__``'s own
    handler fails again on ``skill_id`` while logging the first, so nothing
    readable is ever logged. There is nothing registered to remove in that
    state, so the teardown is a no-op rather than a pair of AttributeErrors.
    """
    return hasattr(obj, "intent_service")


@dataclass
class OCPPlayerProxy:
    """proxy object tracking the state of connected player devices (Sessions)"""
    session_id: str
    available_extractors: List[str]
    ocp_available: bool
    player_state: PlayerState = PlayerState.STOPPED
    media_state: MediaState = MediaState.UNKNOWN
    media_type: MediaType = MediaType.GENERIC
    skill_id: Optional[str] = None


# for easier typing
RawResultsList = List[Union[MediaEntry, Playlist, PluginStream, Dict[str, Any]]]
NormalizedResultsList = List[Union[MediaEntry, Playlist, PluginStream]]


class OCPPipelineMatcher(ConfidenceMatcherPipeline, OVOSAbstractApplication):
    intents = ["play.intent", "open.intent", "media_stop.intent",
               "next.intent", "prev.intent", "pause.intent",
               #  "play_favorites.intent", "like_song.intent",  # handled by ovos-media not ovos-audio, re-enable later
               "resume.intent", "save_game.intent", "load_game.intent"]
    intent_matchers = {}
    intent_cache = f"{xdg_data_home()}/{get_xdg_base()}/intent_cache"
    _voc_cache: Dict[str, List[str]] = {}
    # legacy MediaType labels the loaded in-process MediaProviders serve,
    # the provider-side counterpart of media2skill
    provider_media_types: List[MediaType] = []

    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None):
        """
        Initialize the OCPPipelineMatcher, setting up OCP and legacy audio interfaces, intent and event registration, player session tracking, skill and media mappings, and the AhocorasickNER for entity recognition.
        
        Parameters:
            bus (Optional[Union[MessageBusClient, FakeBus]]): The message bus for event communication. If not provided, a fake bus is used.
            config (Optional[Dict]): Optional configuration dictionary for pipeline and entity keyword setup.
        """
        intent_config = Configuration().get('intents', {})
        config = config or intent_config.get("ovos-ocp-pipeline-plugin") or intent_config.get("OCP") or dict()              
        OVOSAbstractApplication.__init__(
            self, bus=bus or FakeBus(), skill_id=OCP_ID, resources_dir=f"{dirname(__file__)}")
        ConfidenceMatcherPipeline.__init__(self, bus, config)

        self.ocp_api = OCPInterface(self.bus)
        self.legacy_api = ClassicAudioServiceInterface(self.bus)

        self.search_lock = RLock()
        self.ocp_sessions = {}  # session_id: PlaybackCapabilities

        self.skill_aliases = {
            # "skill_id": ["names"]
        }
        self.media2skill = {
            m: [] for m in MediaType
        }
        self.entity_csvs = self.config.get("entity_csvs", [])  # user defined keyword csv files
        self.ner = AhocorasickNER()

        # in-process MediaProvider plugins (opm.media.provider). These run
        # ALONGSIDE the legacy bus-broadcast @ocp_search skill flow (not
        # instead of it) -- both windows execute and their results are merged
        # in `_search` via `_merge_provider_results`. When no providers are
        # installed/loaded this dict stays empty and the pipeline behaves
        # exactly like the pre-existing bus-only path (see `_search_providers`).
        self.media_providers: Dict[str, "MediaProvider"] = {}
        self._load_media_providers()

        # Context-aware bridge to the standalone ovos-media-classifier: builds
        # player_status (now-playing) + ner_list (registered skill keywords) and
        # passes them to its classify_full contract. Defaults to the keyword
        # backend; a config may select a richer opm.media.classifier plugin.
        self.context_classifier = ContextAwareClassifier(
            config=self.config.get("media_classifier", self.config))

        self.register_ocp_api_events()
        self.register_ocp_intents()
        # request available Stream extractor plugins from OCP
        self.bus.emit(Message("ovos.common_play.SEI.get"))

    @classmethod
    def load_resource_files(cls):
        intents = {}
        langs = Configuration().get('secondary_langs', []) + [Configuration().get('lang', "en-US")]
        langs = set([standardize_lang(l) for l in langs])
        for lang in langs:
            lang = standardize_lang(lang)
            intents[lang] = {}
            locale_root = join(dirname(__file__), "locale")
            match = closest_lang(lang, [d for d in os.listdir(locale_root)
                                        if isdir(join(locale_root, d))])
            locale_folder = join(locale_root, match) if match else None
            if locale_folder is not None:
                for f in os.listdir(locale_folder):
                    path = join(locale_folder, f)
                    if f in cls.intents:
                        with open(path) as intent:
                            # a file that ends with a newline, or carries a
                            # blank line, yields an empty sample here.
                            # padatious drops it, but padacioso expands it and
                            # ovos-spec-tools raises MalformedTemplate, which
                            # aborts the matcher for the whole language.
                            samples = [s.replace("{{", "{").replace("}}", "}")
                                       for s in intent.read().split("\n")
                                       if s.strip()]
                            intents[lang][f] = samples
        return intents

    def register_ocp_api_events(self):
        """
        Register messagebus handlers for OCP events
        """
        self.add_event("ovos.common_play.search", self.handle_search_query)
        self.add_event("ovos.common_play.play_search", self.handle_play_search)
        self.add_event('ovos.common_play.status.response', self.handle_player_state_update)
        self.add_event('ovos.common_play.track.state', self.handle_track_state_update)

        self.add_event('ovos.common_play.register_keyword', self.handle_skill_keyword_register)
        self.add_event('ovos.common_play.deregister_keyword', self.handle_skill_keyword_deregister)
        self.add_event('ovos.common_play.announce', self.handle_skill_register)

        self.add_event("mycroft.audio.playing_track", self._handle_legacy_audio_start)
        self.add_event("mycroft.audio.queue_end", self._handle_legacy_audio_end)
        self.add_event("mycroft.audio.service.pause", self._handle_legacy_audio_pause)
        self.add_event("mycroft.audio.service.resume", self._handle_legacy_audio_resume)
        self.add_event("mycroft.audio.service.stop", self._handle_legacy_audio_stop)
        # Launch-time sync: there is no caller and no session yet, so this one
        # is deliberately context-free. A message with no session context
        # resolves to the default session, which is the right session for a
        # sync that belongs to the service rather than to any request.
        self.bus.emit(Message("ovos.common_play.status"))

    @classmethod
    def load_intent_files(cls):
        intent_files = cls.load_resource_files()

        try:
            from ovos_padatious import IntentContainer
            is_padatious = True
        except ImportError:
            from padacioso import IntentContainer
            is_padatious = False
            LOG.warning("Padatious not available, using padacioso. intent matching will be orders of magnitude slower!")

        for lang, intent_data in intent_files.items():
            lang = standardize_lang(lang)
            if is_padatious:
                cache = f"{cls.intent_cache}/{lang}"
                cls.intent_matchers[lang] = IntentContainer(cache)
            else:
                cls.intent_matchers[lang] = IntentContainer()
            for intent_name in cls.intents:
                samples = intent_data.get(intent_name)
                if samples:
                    LOG.debug(f"registering OCP intent: {intent_name}")
                    cls.intent_matchers[lang].add_intent(
                        intent_name.replace(".intent", ""), samples)
            if is_padatious:
                cls.intent_matchers[lang].train()

    def register_ocp_intents(self):
        self.load_intent_files()
        self.add_event("ocp:play", self.handle_play_intent, is_intent=True)
        self.add_event("ocp:play_favorites", self.handle_play_favorites_intent, is_intent=True)
        self.add_event("ocp:open", self.handle_open_intent, is_intent=True)
        self.add_event("ocp:next", self.handle_next_intent, is_intent=True)
        self.add_event("ocp:prev", self.handle_prev_intent, is_intent=True)
        self.add_event("ocp:pause", self.handle_pause_intent, is_intent=True)
        self.add_event("ocp:resume", self.handle_resume_intent, is_intent=True)
        self.add_event("ocp:media_stop", self.handle_stop_intent, is_intent=True)
        self.add_event("ocp:search_error", self.handle_search_error_intent, is_intent=True)
        self.add_event("ocp:like_song", self.handle_like_intent, is_intent=True)
        self.add_event("ocp:save_game", self.handle_save_intent, is_intent=True)
        self.add_event("ocp:load_game", self.handle_load_intent, is_intent=True)

    def _load_media_providers(self):
        """Discover and instantiate installed ``opm.media.provider`` plugins.

        ``load_media_providers`` instantiates every installed provider that is
        not disabled by the per-provider ``enabled: false`` config gate.
        Runtime availability (missing API key, no network, ...) is the
        provider's own concern -- it simply returns ``[]`` from
        :meth:`MediaProvider.search`. When no providers are installed/enabled,
        ``self.media_providers`` stays empty and ``_search_providers`` is a
        guaranteed no-op: the pipeline behaves exactly as the pre-existing
        bus-only OCP-skill path.

        The whole in-process window has an explicit off-switch: setting
        ``media_providers.enabled`` to ``false`` in the OCP pipeline config
        skips loading entirely (default: enabled).
        """
        cfg = self.config.get("media_providers", None)
        if isinstance(cfg, dict) and cfg.get("enabled") is False:
            # documented off-switch: `media_providers.enabled: false` in the
            # OCP pipeline config disables the in-process window entirely and
            # restores the pure legacy bus-only search.
            LOG.info("in-process MediaProvider dispatch disabled by config "
                     "(media_providers.enabled = false)")
            return
        try:
            self.media_providers = load_media_providers(cfg) or {}
            if self.media_providers:
                LOG.info(f"Loaded {len(self.media_providers)} MediaProvider "
                         f"plugins: {list(self.media_providers)}")
                self.provider_media_types = self._collect_provider_media_types()
                LOG.debug(f"MediaProvider media types: {self.provider_media_types}")
        except Exception:
            LOG.exception("failed to load MediaProvider plugins")
            self.media_providers = {}

    def _collect_provider_media_types(self) -> List[MediaType]:
        """Legacy MediaType labels the loaded providers can serve.

        A provider declares what it serves with a class-level set of
        mediavocab (or legacy) media types; the name is read from
        ``SERVED_MEDIA``/``media_types``/``supported_media_types``, whichever
        the plugin defines. Declaring nothing means the provider does its own
        routing and may answer anything, so it contributes every label -- the
        permissive reading the MediaProvider contract asks for ("a provider
        that cannot serve the query just returns an empty list").
        """
        labels = set()
        for name, provider in self.media_providers.items():
            declared = self._declared_media_types(provider)
            if declared is None:
                LOG.debug(f"MediaProvider '{name}' declares no media types, "
                          f"assuming it can serve any request")
                return list(MediaType)
            labels |= declared
        return [m for m in MediaType if m in labels]

    @staticmethod
    def _declared_media_types(provider) -> Optional[set]:
        """Legacy MediaType labels ``provider`` declares it serves.

        A provider declares them as a collection of mediavocab (or legacy)
        media types under ``SERVED_MEDIA``/``media_types``/
        ``supported_media_types``, whichever the plugin defines. Returns
        ``None`` when it declares nothing: that provider does its own routing
        and may answer anything.
        """
        for attr in ("SERVED_MEDIA", "media_types", "supported_media_types"):
            declared = getattr(provider, attr, None)
            if isinstance(declared, (set, frozenset, list, tuple)) and declared:
                return ocp_media_types_for(declared)
        return None

    def _default_valid_labels(self) -> List[MediaType]:
        """Media types this deployment can actually serve.

        The union of the types legacy OCP skills registered for and the types
        the in-process MediaProviders declare. When neither side says
        anything, nothing is known about routing and every type stays valid.
        """
        labels = [m for m, s in self.media2skill.items() if s]
        labels += [m for m in self.provider_media_types if m not in labels]
        return labels or list(MediaType)

    def update_player_proxy(self, player: OCPPlayerProxy):
        """remember OCP session state"""
        self.ocp_sessions[player.session_id] = player

    def handle_skill_register(self, message: Message):
        """
        Registers a skill's names and aliases as keywords for media type matching.
        
        Associates the skill's aliases with appropriate media type labels in the named entity recognizer, enabling accurate media intent classification and routing. Updates internal mappings of skills to media types and aliases.
        """
        skill_id = message.data["skill_id"]
        media = message.data.get("media_types") or \
                message.data.get("media_type") or []
        has_featured_media = message.data.get("featured_tracks", False)
        thumbnail = message.data.get("thumbnail", "")
        display_name = message.data["skill_name"].replace(" Skill", "")
        aliases = message.data.get("aliases", [display_name])
        LOG.info(f"Registering OCP Keyword for {skill_id} : {aliases}")
        self.skill_aliases[skill_id] = aliases

        for idx, m in enumerate(media):
            try:
                m = self._normalize_media_enum(m)
                self.media2skill[m].append(skill_id)
                media[idx] = m
            except:
                LOG.error(f"{skill_id} reported an invalid media_type: {m}")

        # TODO - review below and add missing
        # set bias in classifier
        # aliases -> {type}_streaming_service bias
        for a in aliases:
            if MediaType.MUSIC in media:
                self.ner.add_word("music_streaming_service", a)
            if MediaType.MOVIE in media:
                self.ner.add_word("movie_streaming_service", a)
            # if MediaType.SILENT_MOVIE in media:
            #    self.ner.add_word("silent_movie_streaming_service", a)
            # if MediaType.BLACK_WHITE_MOVIE in media:
            #    self.ner.add_word("bw_movie_streaming_service", a)
            if MediaType.SHORT_FILM in media:
                self.ner.add_word("shorts_streaming_service", a)
            if MediaType.PODCAST in media:
                self.ner.add_word("podcast_streaming_service", a)
            if MediaType.AUDIOBOOK in media:
                self.ner.add_word("audiobook_streaming_service", a)
            if MediaType.NEWS in media:
                self.ner.add_word("news_provider", a)
            if MediaType.TV in media:
                self.ner.add_word("tv_streaming_service", a)
            if MediaType.RADIO in media:
                self.ner.add_word("radio_streaming_service", a)
            if MediaType.ADULT in media:
                self.ner.add_word("porn_streaming_service", a)

    def handle_skill_keyword_register(self, message: Message):
        """
        Register skill-provided keywords and samples for entity recognition.
        
        Adds keywords from a CSV file and/or provided samples to the named entity recognizer for the specified skill and media type.
        """
        skill_id = message.data["skill_id"]
        kw_label = message.data["label"]
        media = message.data["media_type"]
        samples = message.data.get("samples", [])
        csv_path = message.data.get("csv")

        if csv_path:
            with open(csv_path) as f:
                lines = f.read().split("\n")[1:]
                for l in lines:
                    if not l.strip():
                        continue
                    label, value = l.split(",", 1)
                    self.ner.add_word(label, value)

        for s in samples:
            self.ner.add_word(kw_label, s)


    def handle_skill_keyword_deregister(self, message: Message):
        """
        Placeholder for deregistering skill-provided keywords from the entity recognizer.
        
        Currently not implemented.
        """
        skill_id = message.data["skill_id"]
        kw_label = message.data["label"]
        media = message.data["media_type"]
        # TODO

    def handle_track_state_update(self, message: Message):
        """
        Handles track state update messages and updates the player proxy to reflect active playback when a playing state is detected.
        
        Raises:
            ValueError: If the message does not contain a 'state' field.
        """
        state = message.data.get("state")
        if state is None:
            raise ValueError(f"Got state update message with no state: "
                             f"{message}")
        if isinstance(state, int):
            state = TrackState(state)
        player = self.get_player(message)
        if player.player_state != PlayerState.PLAYING and \
                state in [TrackState.PLAYING_AUDIO, TrackState.PLAYING_AUDIOSERVICE,
                          TrackState.PLAYING_VIDEO, TrackState.PLAYING_WEBVIEW,
                          TrackState.PLAYING_MPRIS]:
            player = self.get_player(message)
            player.player_state = PlayerState.PLAYING
            player = self._update_player_skill_id(player, message)
            LOG.info(f"Session: {player.session_id} OCP PlayerState: PlayerState.PLAYING")
            self.update_player_proxy(player)

    def handle_player_state_update(self, message: Message):
        """
        Handles 'ovos.common_play.status' messages with player status updates
        @param message: Message providing new "state" data
        """
        player = self.get_player(message)
        pstate: int = message.data.get("player_state")
        mstate: int = message.data.get("media_state")
        mtype: int = message.data.get("media_type")
        if pstate is not None:
            player.player_state = PlayerState(pstate)
            LOG.debug(f"Session: {player.session_id} PlayerState: {player.player_state}")
        if mstate is not None:
            player.media_state = MediaState(mstate)
            LOG.debug(f"Session: {player.session_id} MediaState: {player.media_state}")
        if mtype is not None:
            player.media_type = MediaType(mtype)
            LOG.debug(f"Session: {player.session_id} MediaType: {player.media_type}")
        player = self._update_player_skill_id(player, message)
        self.update_player_proxy(player)

    # pipeline
    def match_high(self, utterances: List[str], lang: str, message: Message = None) -> Optional[IntentHandlerMatch]:
        """ exact matches only, handles playback control
        recommended after high confidence intents pipeline stage """

        if not len(self.skill_aliases):  # skill_id registered when skills load
            return None  # dont waste compute cycles, no media skills -> no match

        lang = self._get_closest_lang(lang)
        if lang is None:  # no intents registered for this lang
            return None

        utterance = utterances[0].lower()

        # avoid common confusion with alerts and parrot skill
        if (voc_match(utterance, "Alerts", lang, locale=LOCALE_DIR) or
                voc_match(utterance, "SoundIntents", lang, locale=LOCALE_DIR) or
                voc_match(utterance, "Parrot", lang, locale=LOCALE_DIR)):
            return None

        # OVOS-SESSION: a derived Message carries the context of the message
        # it was derived from. This fires on every media utterance, so a bare
        # Message here would put a session-less message on the bus each time
        # and resolve to the default session instead of the caller's.
        # dig_for_message() is this module's own idiom for recovering the
        # request when a caller did not pass it (see _get_player).
        src = message or dig_for_message()
        self.bus.emit(src.forward("ovos.common_play.status") if src
                      else Message("ovos.common_play.status"))

        match = self.intent_matchers[lang].calc_intent(utterance)

        if hasattr(match, "name"):  # padatious
            match = {
                "name": match.name,
                "conf": match.conf,
                "entities": match.matches
            }

        if match["name"] is None:
            return None

        if match.get("conf", 1.0) < 0.7:
            LOG.debug(f"Ignoring low confidence OCP match: {match}")
            return None

        LOG.info(f"OCP match: {match}")

        player = self.get_player(message)

        if player.media_type == MediaType.GAME:
            # if the user is currently playing a game
            # disable: next/prev/shuffle/... intents
            # enable: load/save intents
            game_blacklist = ["next", "prev", "open", "like_song", "play_favorites"]
            if match["name"] in game_blacklist:
                LOG.info(f'Ignoring OCP intent match {match["name"]}, playing MediaType.GAME')
                return None
        else:
            # if no game is being played, disable game specific intents
            game_only = ["save_game", "load_game"]
            # TODO - allow load_game without being in game already
            #  this can only be done if we match skill_id
            if match["name"] in game_only:
                LOG.info(f'Ignoring OCP intent match {match["name"]}, not playing MediaType.GAME')
                return None

        if match["name"] == "play":
            query = match["entities"].pop("query")
            return self._process_play_query(query, utterance, lang, match)

        if match["name"] == "like_song" and player.media_type != MediaType.MUSIC:
            LOG.debug("Ignoring like_song intent, current media is not MediaType.MUSIC")
            return None

        if match["name"] not in ["open", "play_favorites"] and player.player_state == PlayerState.STOPPED:
            LOG.info(f'Ignoring OCP intent match {match["name"]}, OCP Virtual Player is not active')
            # next / previous / pause / resume not targeted
            # at OCP if playback is not happening / paused
            # TODO - handle resume for last_played query, eg, previous day
            return None

        # a control intent only matches when the player is in a state it can
        # act on (OCP-1 §4.3): pausing requires advancing media, resuming
        # requires held media. When the request could not change state, decline
        # the match so the utterance falls through the pipeline instead of
        # matching here and dead-ending in a no-op handler.
        if match["name"] == "pause" and player.player_state != PlayerState.PLAYING:
            LOG.info(f'Ignoring OCP pause match, nothing to pause (PlayerState: {player.player_state})')
            return None
        if match["name"] == "resume" and player.player_state != PlayerState.PAUSED:
            LOG.info(f'Ignoring OCP resume match, nothing to resume (PlayerState: {player.player_state})')
            return None

        return IntentHandlerMatch(match_type=f'ocp:{match["name"]}',
                                  match_data=match,
                                  skill_id=OCP_ID,
                                  utterance=utterance)

    def _extract_entities(self, utterance: str) -> Dict[str, str]:
        """
        Extract media-related entities from an utterance using the NER.

        Returns an empty dict when no skill keywords/entities have been
        registered yet. ahocorasick refuses to build an automaton from an
        empty trie (raising "Not an Aho-Corasick automaton yet"), so guard
        against that instead of relying on a noisy try/except.
        """
        if not len(self.ner.automaton):
            return {}
        try:
            return {e["label"]: e["word"] for e in self.ner.tag(utterance)}
        except Exception as e:
            LOG.error(f"failed to extract media entities: ({e})")
            return {}

    def match_medium(self, utterances: List[str], lang: str, message: Message = None) -> Optional[IntentHandlerMatch]:
        """
        Performs medium-confidence intent matching for media playback queries using classifiers and entity extraction.
        
        Analyzes the first utterance to determine if it is an OCP (Open Common Play) query, classifies the requested media type, and extracts relevant entities. Returns an `IntentHandlerMatch` with extracted information if a match is found; otherwise, returns `None`.
        
        Returns:
            Optional[IntentHandlerMatch]: An intent match object containing media type, entities, query string, and confidence, or `None` if no match is found.
        """
        lang = standardize_lang(lang)

        utterance = utterances[0].lower()
        # is this a OCP query ?
        is_ocp, bconf = self.is_ocp_query(utterance, lang)

        if not is_ocp:
            return None

        # content filter: blocked content (adult by default) never routes
        blocked, reason = self.is_blocked_content(utterance, lang)
        if blocked:
            LOG.info(f"OCP query blocked by content filter ({reason})")
            return None

        # classify the query media type
        media_type, confidence = self.classify_media(utterance, lang, message=message)

        # extract entities
        ents = self._extract_entities(utterance)

        # extract the query string
        query = self.remove_voc(utterance, "Play", lang).strip()

        # rich provider-ready Signals (lossless multi-axis description) handed to
        # the MediaProviders alongside the legacy media_type
        signals = self.media_signals(query or utterance, lang)

        return IntentHandlerMatch(match_type="ocp:play",
                                  match_data={"media_type": media_type,
                                              "entities": ents,
                                              "query": query,
                                              "is_ocp_conf": bconf,
                                              "conf": confidence,
                                              "signals": signals},
                                  skill_id=OCP_ID,
                                  utterance=utterance)

    def match_low(self, utterances: List[str], lang: str, message: Message = None) -> Optional[IntentHandlerMatch]:
        """
        Perform low-confidence matching of an utterance based on the presence of known OCP media keywords.
        
        Attempts to extract media-related entities from the utterance using the internal NER. If entities are found and the media type classification confidence meets a minimum threshold, returns an intent match for OCP playback; otherwise, returns None.
        
        Returns:
            IntentHandlerMatch: An intent match object if a suitable media keyword is found and classified with sufficient confidence, otherwise None.
        """
        utterance = utterances[0].lower()
        # extract entities
        ents = self._extract_entities(utterance)

        if not ents:
            return None

        lang = standardize_lang(lang)

        # content filter: blocked content (adult by default) never routes
        blocked, reason = self.is_blocked_content(utterance, lang)
        if blocked:
            LOG.info(f"OCP query blocked by content filter ({reason})")
            return None

        # classify the query media type
        media_type, confidence = self.classify_media(utterance, lang, message=message)

        if confidence < 0.3:
            return None

        # extract the query string
        query = self.remove_voc(utterance, "Play", lang).strip()

        # rich provider-ready Signals handed to the MediaProviders
        signals = self.media_signals(query or utterance, lang)

        return IntentHandlerMatch(match_type="ocp:play",
                                  match_data={"media_type": media_type,
                                              "entities": ents,
                                              "query": query,
                                              "conf": float(confidence),
                                              "signals": signals},
                                  skill_id=OCP_ID,
                                  utterance=utterance)

    def _process_play_query(self, query:str, utterance: str, lang: str, match: dict = None,
                            message: Optional[Message] = None) -> Optional[IntentHandlerMatch]:
        """
        Process a play query to determine the appropriate playback action or search intent.
        
        If the query indicates a resume action (e.g., "play" while paused), returns a resume intent. Otherwise, prompts for missing queries, identifies explicitly requested skills, classifies the media type, extracts relevant entities, and constructs an intent match for playback.
        
        Parameters:
            query (str): The user's spoken or typed query.
            utterance (str): The original utterance from the user.
            lang (str): The language code for processing.
            match (dict, optional): Existing match data to include in the result.
            message (Message, optional): The message context for the request.
        
        Returns:
            Optional[IntentHandlerMatch]: An intent match object for playback, resume, or search error, or None if no action is determined.
        """
        lang = standardize_lang(lang)
        match = match or {}
        player = self.get_player(message)
        # if media is currently paused, empty string means "resume playback"
        if player.player_state == PlayerState.PAUSED and \
                self._should_resume(query, lang, message=message):
            return IntentHandlerMatch(match_type="ocp:resume",
                                      match_data=match,
                                      skill_id=OCP_ID,
                                      utterance=utterance)

        if not query:
            # user just said "play", we are missing the search query
            phrase = self.get_response("play.what", num_retries=2)
            if not phrase:
                # let the error intent handler take action
                return IntentHandlerMatch(match_type="ocp:search_error",
                                          match_data=match,
                                          skill_id=OCP_ID,
                                          utterance=utterance)

        sess = SessionManager.get(message)
        # if a skill was explicitly requested, search it first
        valid_skills = [
            skill_id for skill_id, samples in self.skill_aliases.items()
            if skill_id not in sess.blacklisted_skills and
               any(s.lower() in utterance for s in samples)
        ]
        valid_labels = []
        if valid_skills:
            LOG.info(f"OCP specific skill names matched: {valid_skills}")
            for mtype, skills in self.media2skill.items():
                if any([s in skills for s in valid_skills]):
                    valid_labels.append(mtype)

        # classify the query media type
        media_type, conf = self.classify_media(utterance, lang, valid_labels=valid_labels, message=message)

        # remove play verb from the query string
        query = self.remove_voc(query, "Play", lang).strip()

        # extract entities
        ents = self._extract_entities(utterance)

        # rich provider-ready Signals handed to the MediaProviders
        signals = self.media_signals(query or utterance, lang)

        return IntentHandlerMatch(match_type="ocp:play",
                                  match_data={"media_type": media_type,
                                              "query": query,
                                              "entities": ents,
                                              "skills": valid_skills,
                                              "conf": match["conf"],
                                              "media_conf": float(conf),
                                              "signals": signals,
                                              # "results": results,
                                              "lang": lang},
                                  skill_id=OCP_ID,
                                  utterance=utterance)

    # bus api
    def handle_search_query(self, message: Message):
        utterance = message.data["utterance"].lower()
        phrase = message.data.get("query", "") or utterance
        # A message with no session context is not an English request: it is
        # one that named no session, and OVOS-SESSION resolves it to the
        # default session, which carries this box's configured language.
        lang = message.data.get("lang") or SessionManager.get(message).lang
        LOG.debug(f"Handle {message.msg_type} request: {phrase}")
        num = message.data.get("number", "")
        if num:
            phrase += " " + num

        lang = standardize_lang(lang)
        # classify the query media type
        media_type, prob = self.classify_media(utterance, lang, message=message)
        # search common play skills
        results = self._search(phrase, media_type, lang, message=message)
        best = self.select_best(results, message)
        results = [r.as_dict if isinstance(best, (MediaEntry, Playlist)) else r
                   for r in results]
        if isinstance(best, (MediaEntry, Playlist)):
            best = best.as_dict
        self.bus.emit(message.response(data={"results": results,
                                             "best": best,
                                             "media_type_conf": float(prob)}))

    def handle_play_search(self, message: Message):
        LOG.info("searching and playing best OCP result")
        utterance = message.data["utterance"].lower()
        query = utterance
        match = self._process_play_query(query, utterance, self.lang, {"conf": 1.0})
        self.bus.emit(message.forward(match.match_type, match.match_data))

    def handle_play_favorites_intent(self, message: Message):
        LOG.info("playing favorite tracks")
        self.bus.emit(message.forward("ovos.common_play.liked_tracks.play"))

    # intent handlers
    @staticmethod
    def _normalize_media_enum(m: Union[int, MediaType]):
        if isinstance(m, MediaType):
            return m
        # convert int to enum
        for e in MediaType:
            if e == m:
                return e
        raise ValueError(f"{m} is not a valid media type")

    def handle_save_intent(self, message: Message):
        skill_id = self.get_player(message).skill_id
        self.bus.emit(message.forward(f"ovos.common_play.{skill_id}.save"))

    def handle_load_intent(self, message: Message):
        skill_id = self.get_player(message).skill_id
        self.bus.emit(message.forward(f"ovos.common_play.{skill_id}.load"))

    def handle_play_intent(self, message: Message):

        if not len(self.skill_aliases) and not len(self.media_providers):
            # skill_id registered when skills load, in-process MediaProviders
            # live in self.media_providers instead
            self.speak_dialog("no.media.skills")
            return

        self.speak_dialog("just.one.moment")

        lang = message.data["lang"]
        query = message.data["query"]
        media_type = message.data["media_type"]
        skills = message.data.get("skills", [])
        sess = SessionManager.get(message)

        # search common play skills
        lang = standardize_lang(lang)
        results = self._search(query, media_type, lang,
                               skills=skills, message=message)

        # tell OCP to play
        self.bus.emit(message.forward('ovos.common_play.reset'))
        if not results:
            self.speak_dialog("cant.play",
                              data={"phrase": query,
                                    "media_type": media_type})
        else:
            LOG.debug(f"Playing {len(results)} results for: {query}")
            best = self.select_best(results, message)
            if best is None:
                self.speak_dialog("cant.play",
                                  data={"phrase": query,
                                        "media_type": media_type})
                return
            LOG.debug(f"OCP Best match: {best}")
            results = [r for r in results if r.as_dict != best.as_dict]
            results.insert(0, best)
            self.set_context("Playing", origin=OCP_ID)

            # ovos-PHAL-plugin-mk1 will display music icon in response to play message
            player = self.get_player(message)
            player.skill_id = best.skill_id
            player.player_state = PlayerState.PLAYING
            player.media_type = best.media_type
            self.update_player_proxy(player)
            # add active skill to session. A result from an in-process
            # MediaProvider carries the provider's name as skill_id; that is
            # no skill on the bus, so nothing would answer its stop ping or
            # its "<skill_id>:stop" - OCP is what plays it, so OCP is the
            # active skill then
            if best.skill_id in self.media_providers:
                sess.activate_skill(OCP_ID)
            else:
                sess.activate_skill(best.skill_id)
            message.context["session"] = sess.serialize()
            if not player.ocp_available:
                self.legacy_play(results, query, message=message)
            else:
                self.ocp_api.play(tracks=[best], utterance=query, source_message=message)
            self.ocp_api.populate_search_results(tracks=results,
                                                 replace=True,
                                                 sort_by_conf=False,  # already sorted
                                                 source_message=message)

    def handle_open_intent(self, message: Message):
        LOG.info("Requesting OCP homescreen")
        # let ovos-media handle it
        self.bus.emit(message.forward('ovos.common_play.home'))

    def handle_like_intent(self, message: Message):
        LOG.info("Requesting OCP to like current song")
        # let ovos-media handle it
        self.bus.emit(message.forward("ovos.common_play.like"))

    def handle_stop_intent(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available:
            LOG.info("Requesting Legacy AudioService to stop")
            self.legacy_api.stop(source_message=message)
        else:
            LOG.info("Requesting OCP to stop")
            self.ocp_api.stop(source_message=message)
        player = self.get_player(message)
        player.player_state = PlayerState.STOPPED
        player.skill_id = None
        self.update_player_proxy(player)

    # the stop pipeline (OVOS-STOP-1): a bare "stop" while OCP plays
    def can_stop(self, message: Message) -> bool:
        """True while OCP is playing or paused in this message's Session,
        so the stop pipeline's ping finds OCP and a bare "stop" stops the
        playback, through ovos-media or the legacy audio service alike."""
        sess = SessionManager.get(message)
        player = self.ocp_sessions.get(sess.session_id)
        return bool(player) and player.player_state in (PlayerState.PLAYING,
                                                        PlayerState.PAUSED)

    def stop_session(self, session: Session) -> bool:
        """Stop what OCP plays in ``session`` (the stop pipeline's
        ``<skill_id>:stop`` / ``stop:skill`` for OCP). Only stop_session is
        implemented, not stop(), which also runs on shutdown."""
        player = self.ocp_sessions.get(session.session_id)
        if not player or player.player_state not in (PlayerState.PLAYING,
                                                     PlayerState.PAUSED):
            return False
        self.handle_stop_intent(Message("ocp:media_stop",
                                        context={"session": session.serialize(),
                                                 "skill_id": OCP_ID}))
        return True

    def handle_next_intent(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available:
            LOG.info("Requesting Legacy AudioService to go to next track")
            self.legacy_api.next(source_message=message)
        else:
            LOG.info("Requesting OCP to go to next track")
            self.ocp_api.next(source_message=message)

    def handle_prev_intent(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available:
            LOG.info("Requesting Legacy AudioService to go to prev track")
            self.legacy_api.prev(source_message=message)
        else:
            LOG.info("Requesting OCP to go to prev track")
            self.ocp_api.prev(source_message=message)

    def handle_pause_intent(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available:
            LOG.info("Requesting Legacy AudioService to pause")
            self.legacy_api.pause(source_message=message)
        else:
            LOG.info("Requesting OCP to go to pause")
            self.ocp_api.pause(source_message=message)
        player = self.get_player(message)
        player.player_state = PlayerState.PAUSED
        player = self._update_player_skill_id(player, message)
        self.update_player_proxy(player)

    def handle_resume_intent(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available:
            LOG.info("Requesting Legacy AudioService to resume")
            self.legacy_api.resume(source_message=message)
        else:
            LOG.info("Requesting OCP to go to resume")
            self.ocp_api.resume(source_message=message)
        player = self.get_player(message)
        player.player_state = PlayerState.PLAYING
        player = self._update_player_skill_id(player, message)
        self.update_player_proxy(player)

    def handle_search_error_intent(self, message: Message):
        self.bus.emit(message.forward("mycroft.audio.play_sound",
                                      {"uri": "snd/error.mp3"}))
        player = self.get_player(message)
        if not player.ocp_available:
            LOG.info("Requesting Legacy AudioService to stop")
            self.legacy_api.stop(source_message=message)
        else:
            LOG.info("Requesting OCP to stop")
            self.ocp_api.stop(source_message=message)

    # NLP
    def _get_context_classifier(self) -> ContextAwareClassifier:
        """Return the context-aware classifier, building it lazily if needed.

        ``__init__`` wires ``self.context_classifier`` from the config; this
        accessor also covers callers that construct the matcher without running
        ``__init__`` (e.g. tests) — it keeps the keyword-backend floor available
        everywhere without a hard dependency on init order.
        """
        clf = getattr(self, "context_classifier", None)
        if clf is None:
            cfg = getattr(self, "config", None) or {}
            clf = ContextAwareClassifier(config=cfg.get("media_classifier", cfg))
            self.context_classifier = clf
        return clf

    def _classifier_context(self, message: Optional[Message] = None):
        """Build the classifier's (player_status, ner_list) context.

        ``player_status`` (a mediavocab ``PlayerStatus``) comes from the per
        -session now-playing proxy and ``ner_list`` (``{label: [entity]}``) from
        the skill-registered keywords already in the NER — the two minimal
        context inputs the standalone ``classify_full`` contract accepts.  These
        let the classifier resolve relative control follow-ups ("next", "pause",
        "something else") and route by entity.
        """
        try:
            player = self.get_player(message, timeout=0)
        except Exception:  # noqa: BLE001 - context is best-effort
            player = None
        return build_player_status(player), build_ner_list(self.ner)

    def media_signals(self, query: str, lang: str) -> Optional[dict]:
        """Build the provider-ready ``mediavocab.Signals`` for *query*.

        Returns the classifier's lossless multi-axis description (medium /
        playback_type / content_genres / content_form / programme_format /
        variant_kind / accessibility / picture_format) as a plain dict to hand to
        the MediaProviders alongside ``media_type``.  ``None`` on any failure so
        the legacy ``(media_type, conf)`` path is never broken.
        """
        try:
            signals = self._get_context_classifier().to_signals(query, lang)
        except Exception as e:  # noqa: BLE001 - signals are additive, never fatal
            LOG.debug(f"could not build media Signals: {e}")
            return None
        for dumper in ("model_dump", "dict"):
            fn = getattr(signals, dumper, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:  # noqa: BLE001
                    pass
        return None

    def is_blocked_content(self, query: str, lang: str) -> Tuple[bool, str]:
        """Apply the classifier's content filter → ``(blocked, reason)``.

        Adult content is blocked by default; configurable via
        ``allow_adult_content`` / ``media_content_filter``.  Blocked queries are
        not routed to providers.
        """
        try:
            return self._get_context_classifier().is_blocked(query, lang)
        except Exception as e:  # noqa: BLE001 - filtering is best-effort, fail open
            LOG.debug(f"content filter unavailable: {e}")
            return False, ""

    def classify_media(self, query: str, lang: str,
                       valid_labels: Optional[List[MediaType]] = None,
                       message: Optional[Message] = None) -> Tuple[MediaType, float]:
        """Determine what (legacy) media type is being requested.

        Backed by the standalone ``ovos-media-classifier`` (the keyword backend
        by default; a config may select a richer ``opm.media.classifier`` plugin
        / ONNX / embedding-router backend — see :class:`ContextAwareClassifier`).
        The classifier speaks ``mediavocab``; this method threads the per-session
        player + NER context, runs the full multi-axis context-aware
        classification, and folds the result back onto the legacy
        ``ovos_utils.ocp.MediaType`` the pipeline emits downstream.

        The classifier abstains (GENERIC) when unsure, so non-media queries are
        never hijacked; the abstain→GENERIC floor is preserved.
        """
        lang = standardize_lang(lang)
        valid_labels = valid_labels or self._default_valid_labels()
        LOG.debug(f"valid media types: {valid_labels}")
        if valid_labels == [MediaType.GENERIC]:
            # GENERIC is not a media type a request can be classified as, it is
            # the "no type" label; a deployment that only registered GENERIC
            # (ovos-media's favorites skill does exactly that) has said nothing
            # about routing. Classify against the full taxonomy instead and let
            # the classifier abstain to GENERIC when nothing matches.
            valid_labels = list(MediaType)
        if len(valid_labels) == 1:
            # only one thing can be served, no point classifying
            return valid_labels[0], 1.0

        # Consult the standalone context-aware classifier: it sees the
        # now-playing state (relative follow-ups) + the registered entities, runs
        # the full multi-axis classification, and folds the leaf + axes back onto
        # the legacy MediaType. It abstains (GENERIC) when unsure, so the
        # abstain→GENERIC floor never hijacks a non-media query.
        #
        # ``valid_labels`` gating (with the axis-refined-leaf -> parent ->
        # broad VIDEO/AUDIO fallback chain) happens inside
        # ``ContextAwareClassifier.classify`` itself, via the shared
        # ``mv_to_legacy_candidates`` — a skill that only registered the
        # coarser parent stays reachable even when the classifier resolves a
        # more specific leaf.
        try:
            player_status, ner_list = self._classifier_context(message)
            media_type, confidence = self._get_context_classifier().classify(
                query, lang, player_status=player_status, ner_list=ner_list,
                valid_labels=valid_labels)
        except Exception as e:  # noqa: BLE001 - never break matching on the bridge
            LOG.debug(f"context classifier unavailable: {e}")
            return MediaType.GENERIC, 0.0

        return media_type, float(confidence)

    def is_ocp_query(self, query: str, lang: str) -> Tuple[bool, float]:
        """Determine if a playback question is being asked.

        Contract preserved: a non-GENERIC type from the classifier-backed
        :meth:`classify_media` means this is an OCP query.
        """
        lang = standardize_lang(lang)
        m, p = self.classify_media(query, lang)
        return m != MediaType.GENERIC, p

    def _should_resume(self, phrase: str, lang: str, message: Optional[Message] = None) -> bool:
        """
        Check if a "play" request should resume playback or be handled as a new
        session.
        @param phrase: Extracted playback phrase
        @return: True if player should resume, False if this is a new request
        """
        lang = standardize_lang(lang)
        player = self.get_player(message)
        if player.player_state == PlayerState.PAUSED:
            if not phrase.strip() or \
                    voc_match(phrase, "Resume", lang=lang, exact=True, locale=LOCALE_DIR) or \
                    voc_match(phrase, "Play", lang=lang, exact=True, locale=LOCALE_DIR):
                return True
        return False

    # search
    def _player_sync(self, player: OCPPlayerProxy, message: Optional[Message] = None, timeout=1) -> OCPPlayerProxy:

        if not self.config.get("legacy"):  # force legacy audio in config
            ev = threading.Event()

            def handle_m(m):
                nonlocal player
                s = SessionManager.get(m)
                if s.session_id == player.session_id:
                    player.available_extractors = m.data["SEI"]
                    player.ocp_available = True
                    self.update_player_proxy(player)
                    ev.set()
                    LOG.debug(f"Session: {player.session_id} Available stream extractor plugins: {m.data['SEI']}")

            self.bus.on("ovos.common_play.SEI.get.response", handle_m)
            message = message or dig_for_message() or Message("")  # get message.context to forward
            self.bus.emit(message.forward("ovos.common_play.SEI.get"))
            ev.wait(timeout)
            self.bus.remove("ovos.common_play.SEI.get.response", handle_m)

            if not ev.is_set():
                LOG.warning(f"Player synchronization timed out after {timeout} seconds")

        return player

    def get_player(self, message: Optional[Message] = None, timeout=1) -> OCPPlayerProxy:
        """get a PlayerProxy object, containing info such as player state and the available stream extractors from OCP
        this is tracked per Session, if needed requests the info from the client"""
        sess = SessionManager.get(message)
        if sess.session_id not in self.ocp_sessions:
            player = OCPPlayerProxy(available_extractors=available_extractors(),
                                    ocp_available=False,
                                    session_id=sess.session_id)
            self.update_player_proxy(player)
        else:
            player = self.ocp_sessions[sess.session_id]
        if not player.ocp_available and not self.config.get("legacy"):
            # OCP might have loaded meanwhile
            player = self._player_sync(player, message, timeout)
        return player

    @staticmethod
    def _update_player_skill_id(player, message):
        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        if skill_id and skill_id != OCP_ID:
            player.skill_id = skill_id
        return player

    @staticmethod
    def normalize_results(results: RawResultsList) -> NormalizedResultsList:
        # support Playlist and MediaEntry objects in tracks
        for idx, track in enumerate(results):
            if isinstance(track, dict):
                try:
                    results[idx] = dict2entry(track)
                except Exception as e:
                    LOG.error(f"got an invalid track: {track}")
                    results[idx] = None
        return [r for r in results if r]

    def filter_results(self, results: list, phrase: str, lang: str,
                       media_type: MediaType = MediaType.GENERIC,
                       message: Optional[Message] = None) -> list:
        lang = standardize_lang(lang)
        # ignore very low score matches
        l1 = len(results)
        results = [r for r in results
                   if r.match_confidence >= self.config.get("min_score", 50)]
        LOG.debug(f"filtered {l1 - len(results)} low confidence results")

        # filter based on MediaType
        if self.config.get("filter_media", True) and media_type != MediaType.GENERIC:
            l1 = len(results)
            # TODO - also check inside playlists
            results = [r for r in results
                       if isinstance(r, Playlist) or r.media_type == media_type]
            LOG.debug(f"filtered {l1 - len(results)} wrong MediaType results")

        # filter based on available stream extractors
        player = self.get_player(message)
        valid_starts = ["/", "http://", "https://", "file://"] + \
                       [f"{sei}//" for sei in player.available_extractors]
        if self.config.get("filter_SEI", True):
            # TODO - also check inside playlists
            bad_seis = [r for r in results if isinstance(r, MediaEntry) and
                        not any(r.uri.startswith(sei) for sei in valid_starts)]

            results = [r for r in results if r not in bad_seis]
            plugs = set([s.uri.split('//')[0] for s in bad_seis if '//' in s.uri])
            if bad_seis:
                LOG.debug(f"filtered {len(bad_seis)} results that require "
                          f"unavailable plugins: {plugs}")

        # filter by media type
        audio_only = voc_match(phrase, "audio_only", lang=lang, locale=LOCALE_DIR)
        video_only = voc_match(phrase, "video_only", lang=lang, locale=LOCALE_DIR)
        if self.config.get("playback_mode") == PlaybackMode.VIDEO_ONLY:
            # select only from VIDEO results if preference is set
            audio_only = True
        elif self.config.get("playback_mode") == PlaybackMode.AUDIO_ONLY:
            # select only from AUDIO results if preference is set
            video_only = True

        # check if user said "play XXX audio only"
        if audio_only or not player.ocp_available:
            l1 = len(results)
            # TODO - also check inside playlists
            results = [r for r in results
                       if (isinstance(r, Playlist) and player.ocp_available)
                       or r.playback == PlaybackType.AUDIO]
            LOG.debug(f"filtered {l1 - len(results)} non-audio results")

        # check if user said "play XXX video only"
        elif video_only:
            l1 = len(results)
            results = [r for r in results
                       if isinstance(r, Playlist) or r.playback == PlaybackType.VIDEO]
            LOG.debug(f"filtered {l1 - len(results)} non-video results")

        return results

    @classmethod
    def _voc_words(cls, lang: str) -> List[str]:
        """Every media-keyword and filler word shipped for ``lang``.

        Read through ``ovos_spec_tools``' OVOS-INTENT-2 common reader, longest
        first, so a phrase can be tested for carrying nothing but vocabulary.
        Missing resources simply yield fewer words.
        """
        lang = standardize_lang(lang)
        if lang in cls._voc_cache:
            return cls._voc_cache[lang]
        words = set()
        # media-type keywords live with ovos-media-classifier; the play verbs
        # and fillers ship here
        from ovos_media_classifier.keyword import _LOCALE_DIR as CLF_LOCALE_DIR
        for root in (LOCALE_DIR, CLF_LOCALE_DIR):
            match = closest_lang(lang, [d for d in os.listdir(root)
                                        if isdir(join(root, d))])
            if not match:
                continue
            folder = join(root, match)
            for f in os.listdir(folder):
                if not (f.endswith("Keyword.voc") or f in ("Play.voc", "Filler.voc")):
                    continue
                # the OVOS-INTENT-2 common reader, not a hand-rolled one:
                # it discards a BOM, accepts CRLF, and drops blank and
                # #-comment lines (section 3)
                words |= {t.lower()
                          for t in read_resource_file(Path(join(folder, f)))}
        cls._voc_cache[lang] = sorted(words, key=len, reverse=True)
        return cls._voc_cache[lang]

    def _is_bare_media_request(self, phrase: str, lang: str) -> bool:
        """True when ``phrase`` names a media type and nothing else.

        "play some music" asks for music, it does not ask for a track called
        "some music". Forwarding that as a title makes every provider fuzzy
        match it against real titles and score too low to survive the
        confidence floor, while the provider contract browses the catalog when
        the title is empty. Detected conservatively: strip the media keywords
        and fillers this plugin ships for the language, with the spec tools'
        whole-word stripper, and see if anything is left. A real title always
        leaves a remainder.
        """
        phrase = (phrase or "").strip()
        if not phrase:
            return True
        # strip_samples anchors on whole words and removes the longest sample
        # first, so "music" inside "music box" is not taken on its own
        return not strip_samples(phrase, self._voc_words(lang)).strip()

    def _provider_signals(self, phrase: str, media_type: MediaType,
                          lang: str):
        """Provider-ready query ``Signals`` for a classified request."""
        title = "" if self._is_bare_media_request(phrase, lang) else phrase
        if not title and phrase:
            LOG.debug(f"'{phrase}' is a bare {media_type} request, asking "
                      f"MediaProviders to browse instead of match a title")
        return media_type_to_signals(media_type, title)

    def _build_query_context(self, lang: str,
                             message: Optional[Message] = None) -> dict:
        """Build the request-context kwargs for the MediaProvider ``search``
        contract, from the session/config.

        The MediaProvider contract is a single ``search(signals, lang="en-us",
        *, supported_playback_types, blocked_genres, region, session_id)``
        call: there is no routing/gating API. The pipeline passes what it
        knows about the request as explicit kwargs; each provider reads the
        keys it cares about and self-filters (returning ``[]`` when it cannot
        serve the query/context).

        Returns the four context kwargs (all permissive by default):
          * ``supported_playback_types`` -- e.g. ``{"audio", "video"}``;
            empty => no device gate. Read from
            ``media.supported_playback_types``.
          * ``blocked_genres`` -- genre tags the content policy blocks, from
            ``media_content_filter.blocked_genres`` plus
            ``media_content_filter.allow_adult_content`` (default ``false``
            => ``adult`` blocked).
          * ``region`` -- ISO 3166-1 alpha-2 from
            ``location.city.region.country.code``.
          * ``session_id`` -- originating session id (from ``message`` when
            given).

        All four are **device/user** settings, not OCP pipeline settings:
        they are read from the global :class:`ovos_config.Configuration`, not
        from ``self.config`` (which is only the ``ocp`` pipeline sub-config
        and never carries these keys).
        """
        config = Configuration()

        media_cfg = config.get("media", {}) or {}
        supported = set(media_cfg.get("supported_playback_types", []) or [])

        cf = config.get("media_content_filter", {}) or {}
        blocked = set(cf.get("blocked_genres", ["adult"]))
        if config.get("allow_adult_content", cf.get("allow_adult_content", False)):
            blocked.discard("adult")

        region = None
        loc = config.get("location", {}) or {}
        code = (((loc.get("city") or {}).get("region") or {}).get("country") or {}).get("code")
        if code:
            region = str(code)

        session_id = None
        if message is not None:
            try:
                session_id = SessionManager.get(message).session_id
            except Exception:
                session_id = None

        return {
            "supported_playback_types": {str(p) for p in supported},
            "blocked_genres": {str(g) for g in blocked},
            "region": region,
            "session_id": session_id,
        }

    @staticmethod
    def _safe_search(provider, signals, lang, **context):
        """Call ``provider.search`` so one provider raising cannot abort the
        multi-provider search, and cannot empty out the legacy bus results it
        runs alongside. Returns ``[]`` on any exception. ``**context`` carries
        the explicit MediaProvider kwargs (``supported_playback_types``,
        ``blocked_genres``, ``region``, ``session_id``)."""
        try:
            return provider.search(signals, lang=lang, **context) or []
        except Exception:
            LOG.exception(f"MediaProvider '{getattr(provider, 'name', provider)}' "
                          f"search failed")
            return []

    def _search_providers(self, phrase: str, media_type: MediaType,
                          lang: str,
                          message: Optional[Message] = None) -> RawResultsList:
        """Dispatch the query to in-process MediaProvider plugins.

        This is the second window of the dual-window search: it runs
        alongside (not instead of) the legacy bus @ocp_search flow in
        :meth:`_execute_query`. Builds a :class:`mediavocab.Signals` from the
        classified media type and query, then runs every loaded provider
        concurrently through a thread pool calling :meth:`_safe_search`
        (which never raises -- a provider exception is caught/logged and
        contributes no results, it cannot empty out the legacy window). Each
        provider receives the query ``Signals``, ``lang`` and the
        request-context kwargs from :meth:`_build_query_context`; a provider
        that cannot serve the query/context returns ``[]``. Each returned
        ``mediavocab.Release`` is bridged to the OCP playback result dict and
        added to the result pool by :meth:`_merge_provider_results`.

        Providers blacklisted for the Session (``blacklisted_skills``) are
        not dispatched to at all: a provider's registry name IS its
        ``skill_id`` downstream, so the same Session gate that suppresses a
        bus skill suppresses the provider of that name.

        The whole dispatch is bounded by the pipeline's ``max_timeout``
        (the same budget the legacy bus window uses): providers still running
        when it expires are cancelled and contribute nothing, so one slow
        provider cannot stall the search.

        Returns an empty list (a guaranteed no-op) when no providers are
        installed/enabled -- this is what keeps the zero-providers case
        bit-identical to the legacy-only behaviour.

        Note: the timeout path calls ``executor.shutdown(wait=False,
        cancel_futures=True)``, which cancels only futures that have not
        started running. A provider thread already executing when the
        timeout fires cannot be cancelled and keeps running to completion
        (or forever, if it hangs) -- a genuinely hanging provider leaks one
        worker thread per hung call for the life of the process. ``wait=False``
        is still correct here: blocking on shutdown would re-join that same
        stuck thread and stall every future search, which is strictly worse
        than a bounded thread leak.
        """
        if not self.media_providers:
            return []

        targets = dict(self.media_providers)
        if message is not None:
            try:
                sess = SessionManager.get(message)
                blacklist = set(sess.blacklisted_skills or [])
            except Exception:
                blacklist = set()
            if blacklist:
                skipped = [n for n in targets if n in blacklist]
                for n in skipped:
                    targets.pop(n)
                if skipped:
                    LOG.debug(f"MediaProviders blacklisted by Session: {skipped}")
        if not targets:
            return []

        media_type = self._normalize_media_enum(media_type)
        if media_type != MediaType.GENERIC:
            # a provider that declared what it serves is only asked about the
            # types it declared. A provider that declared nothing is still
            # asked (it does its own routing), but its answers are ranked
            # last -- see `_demote_undeclared_providers`.
            for name, provider in list(targets.items()):
                declared = self._declared_media_types(provider)
                if declared is not None and media_type not in declared:
                    LOG.debug(f"MediaProvider '{name}' does not serve "
                              f"{media_type} queries, not dispatching")
                    targets.pop(name)
            if not targets:
                return []

        # Build a minimal provider-ready Signals from the classified media
        # type and the free-text query via the local bridge.
        signals = self._provider_signals(phrase, media_type, lang)
        context = self._build_query_context(lang, message=message)

        LOG.debug(f"dispatching to {len(targets)} MediaProviders: {list(targets)}")
        max_timeout = self.config.get("max_timeout", 15)

        results: RawResultsList = []
        executor = ThreadPoolExecutor(max_workers=len(targets))
        try:
            futures = {
                executor.submit(self._safe_search, p, signals, lang, **context): name
                for name, p in targets.items()
            }
            try:
                for fut in as_completed(futures, timeout=max_timeout):
                    name = futures[fut]
                    try:
                        releases = fut.result() or []
                    except Exception:
                        LOG.exception(f"MediaProvider '{name}' search failed")
                        continue
                    if len(releases) > MAX_PROVIDER_RESULTS:
                        LOG.debug(f"MediaProvider '{name}' returned "
                                  f"{len(releases)} results, truncating to "
                                  f"{MAX_PROVIDER_RESULTS}")
                        releases = releases[:MAX_PROVIDER_RESULTS]
                    for release in releases:
                        try:
                            # media_type: stamp the QUERY's legacy media type,
                            # not the fold of the Release's own -- the fold is
                            # not injective (NEWS -> RADIO -> RADIO) and
                            # filter_results would drop the result.
                            results.append(release_to_ocp_result(
                                release, name, media_type=media_type,
                                signals=signals))
                        except Exception:
                            LOG.exception(f"failed to bridge result from '{name}'")
            except FuturesTimeoutError:
                pending = [n for f, n in futures.items() if not f.done()]
                LOG.warning(f"MediaProvider search timed out after "
                            f"{max_timeout}s, dropping: {pending}")
        finally:
            # do NOT use the `with` block: its __exit__ joins every worker
            # thread, which would re-introduce the very stall the timeout
            # exists to prevent.
            executor.shutdown(wait=False, cancel_futures=True)
        LOG.debug(f"MediaProviders returned {len(results)} results")
        return results

    @staticmethod
    def _canonical_uri(uri) -> Optional[str]:
        """Canonical form of a URI, for cross-window de-duplication only.

        Normalizes away the differences that make the same stream look like
        two: ``http`` vs ``https``, host case, and trailing slashes. Anything
        else (query string, path case, port) is left alone -- it can select a
        different resource.
        """
        if not uri:
            return None
        raw = str(uri).strip()
        if not raw:
            return None
        try:
            parts = urlsplit(raw)
        except ValueError:
            return raw.rstrip("/")
        if not parts.scheme:
            return raw.rstrip("/") or None
        scheme = parts.scheme.lower()
        if scheme == "https":
            # http/https is not a different resource for de-dup purposes
            scheme = "http"
        return urlunsplit((scheme, parts.netloc.casefold(),
                           parts.path.rstrip("/"), parts.query, parts.fragment))

    @staticmethod
    def _carries_playlist(item) -> bool:
        """True when a result is (or carries) a playlist rather than a single
        stream. Playlists are never de-duplicated: they aggregate tracks, so a
        uri match says nothing about the entries being equivalent."""
        if isinstance(item, Playlist):
            return True
        if isinstance(item, dict):
            return bool(item.get("playlist") or item.get("tracks"))
        return bool(getattr(item, "playlist", None) or getattr(item, "tracks", None))

    @classmethod
    def _merge_provider_results(cls, bus_results: list,
                                provider_results: list) -> list:
        """Add the in-process MediaProvider window to the legacy bus window.

        Dual-window means BOTH windows contribute. A provider result is
        **added** to the result pool; it never replaces or deletes a legacy
        bus entry. The only de-duplication is in the provider -> pool
        direction: a provider entry is dropped when a legacy entry already
        covers the same canonical ``uri`` (see :meth:`_canonical_uri`).

        There is deliberately **no** ``(title, artist)`` de-duplication tier:
        real data falsifies it (a SomaFM bus skill answers artist ``SomaFM``
        with a stream uri while the provider answers artist ``""`` with a
        playlist uri -- nothing legitimately collapses), and distinct Releases
        of one Work are meant to coexist. Entries carrying a playlist are
        never de-duplicated at all.

        Ranking between the surviving entries is NOT decided here: the
        existing ``normalize_results`` -> ``filter_results`` -> ``select_best``
        pipeline arbitrates provider and bus answers exactly as it already
        arbitrates two competing skills.

        When ``provider_results`` is empty this is a no-op returning
        ``bus_results`` unchanged -- callers should skip calling it entirely
        in that case (see :meth:`_search`), so the zero-providers path never
        even constructs a new list.
        """
        def _get(item, key):
            return item.get(key) if isinstance(item, dict) else getattr(item, key, None)

        merged = list(bus_results)
        legacy_uris = set()
        for item in bus_results:
            if cls._carries_playlist(item):
                continue
            canon = cls._canonical_uri(_get(item, "uri"))
            if canon:
                legacy_uris.add(canon)

        for prov_item in provider_results:
            if not cls._carries_playlist(prov_item):
                canon = cls._canonical_uri(_get(prov_item, "uri"))
                if canon and canon in legacy_uris:
                    LOG.debug(f"dropping provider result already covered by a "
                              f"legacy result: {canon}")
                    continue
            merged.append(prov_item)
        return merged

    @staticmethod
    def _result_field(item, key):
        """Read ``key`` off a result that may be a dict or a MediaEntry."""
        return item.get(key) if isinstance(item, dict) else getattr(item, key, None)

    def _demote_undeclared_providers(self, results: list,
                                     media_type: MediaType) -> list:
        """Rank answers from providers that never declared the queried type last.

        A MediaProvider that declares the media types it serves is only asked
        about those types (see :meth:`_search_providers`). One that declares
        nothing is still asked about everything, because the contract lets a
        provider route for itself -- but "I route for myself" is not evidence
        that its answer fits the request, and a provider is free to score its
        own results 100. Live evidence: a news provider that declares nothing
        answered a MUSIC request with confidence 100 and won over the local
        music library.

        So, for a request with a CONCRETE media type, a result from a provider
        that did not declare that type is capped at
        :data:`UNDECLARED_PROVIDER_MAX_CONFIDENCE` (the default ``min_score``
        floor), and one point lower when some source that does claim the type
        -- a legacy OCP skill or a declaring provider -- also answered at or
        above that cap. The undeclared answer therefore still plays when it is
        the only thing on offer, and never wins against an answer that claims
        the type.

        A GENERIC request ("play something") asks for no type at all, so no
        provider can be off-topic for it and nothing is capped: the existing
        arbitration decides, exactly as before.
        """
        media_type = self._normalize_media_enum(media_type)
        if media_type == MediaType.GENERIC:
            return results
        undeclared = {name for name, provider in self.media_providers.items()
                      if self._declared_media_types(provider) is None}
        if not undeclared:
            return results

        cap = UNDECLARED_PROVIDER_MAX_CONFIDENCE
        if any(self._result_field(r, "skill_id") not in undeclared and
               (self._result_field(r, "match_confidence") or 0) >= cap
               for r in results):
            cap -= 1

        for r in results:
            if self._result_field(r, "skill_id") not in undeclared:
                continue
            if (self._result_field(r, "match_confidence") or 0) <= cap:
                continue
            LOG.debug(f"capping result from '{self._result_field(r, 'skill_id')}' "
                      f"at {cap}: it does not declare {media_type}")
            if isinstance(r, dict):
                r["match_confidence"] = cap
            else:
                r.match_confidence = cap
        return results

    def _search(self, phrase: str, media_type: MediaType, lang: str,
                skills: Optional[List[str]] = None,
                message: Optional[Message] = None) -> list:
        self.bus.emit(message.reply("ovos.common_play.search.start"))
        self.enclosure.mouth_think()  # animate mk1 mouth during search

        # In-process MediaProvider window: runs ALONGSIDE the legacy bus
        # window, never instead of it, and CONCURRENTLY with it -- the two
        # windows are a merge of two independent sources, so the search costs
        # the slower window, not the sum of both. It is started first and
        # collected after the bus window; each window keeps its own timeout.
        # Skill-targeted searches (user named a specific OCP skill) stay
        # bus-only. When no providers are installed/loaded,
        # `_search_providers` is a guaranteed no-op and
        # `_merge_provider_results` is skipped entirely -- `results` is
        # exactly the legacy bus list, unmodified, so output is bit-identical
        # to the pre-existing legacy-only behaviour.
        provider_window = None
        if not skills:
            provider_window = ThreadPoolExecutor(max_workers=1)
            provider_future = provider_window.submit(
                self._search_providers, phrase, media_type, lang,
                message=message)

        # Legacy window: place a query on the messagebus for anyone who wants
        # to attempt to service a 'play.request' message.
        results = []
        try:
            for r in self._execute_query(phrase,
                                         media_type=media_type,
                                         skills=skills,
                                         message=message):
                results += r["results"]

            # Merge, once both windows are done. Provider results are ADDED to
            # the pool -- a legacy result is never replaced or deleted.
            if provider_window is not None:
                provider_results = provider_future.result()
                if provider_results:
                    results = self._merge_provider_results(results,
                                                           provider_results)
        finally:
            if provider_window is not None:
                provider_window.shutdown(wait=False)

        results = self._demote_undeclared_providers(results, media_type)
        results = self.normalize_results(results)

        if not skills:
            LOG.debug(f"Got {len(results)} results")
            results = self.filter_results(results, phrase, lang, media_type,
                                          message=message)
            LOG.debug(f"Got {len(results)} usable results")
        else:  # no filtering if skill explicitly requested
            LOG.debug(f"Got {len(results)} usable results from {skills}")

        self.bus.emit(message.reply("ovos.common_play.search.end"))
        return results

    def _execute_query(self, phrase: str,
                       media_type: MediaType = Union[int, MediaType],
                       skills: Optional[List[str]] = None,
                       message: Optional[Message] = None) -> list:
        """ actually send the search to OCP skills"""
        media_type = self._normalize_media_enum(media_type)

        if not self.skill_aliases:
            # No OCP skill ever registered, so the bus window has no possible
            # responder and every query.wait() would burn the full search
            # timeout as dead air before the provider window even starts.
            LOG.debug("no OCP skills registered, skipping the legacy bus "
                      "search window")
            return []

        with self.search_lock:
            # stop any search still happening
            self.bus.emit(message.reply("ovos.common_play.search.stop"))

            query = OCPQuery(query=phrase, media_type=media_type,
                             config=self.config, bus=self.bus)
            # search individual skills first if user specifically asked for it
            results = []
            if skills:
                for skill_id in skills:
                    if skill_id not in self.media2skill[media_type]:
                        LOG.debug(f"{skill_id} can't handle {media_type} queries")
                        continue
                    LOG.debug(f"Searching OCP Skill: {skill_id}")
                    query.send(skill_id, source_message=message)
                    query.wait()
                    results += query.results

            if not len(self.media2skill[media_type]) and \
                    media_type not in self.provider_media_types:
                LOG.info(f"No skills or MediaProviders available to handle "
                         f"{media_type} queries, forcing MediaType.GENERIC")
                media_type = MediaType.GENERIC

            # search all skills
            if not results:
                if skills:
                    LOG.info(f"No specific skill results from {skills}, "
                             f"performing global OCP search")
                query.reset()
                query.send()
                query.wait()
                results = query.results

            # fallback to generic search type
            if not results and \
                    self.config.get("search_fallback", True) and \
                    media_type != MediaType.GENERIC:
                LOG.debug("OVOSCommonPlay falling back to MediaType.GENERIC")
                query.media_type = MediaType.GENERIC
                query.reset()
                query.send()
                query.wait()
                results = query.results

        LOG.debug(f'Returning {len(results)} search results')
        return results

    @staticmethod
    def select_best(results: list, message: Message) -> Union[MediaEntry, Playlist, PluginStream]:

        sess = SessionManager.get(message)

        # Look at any replies that arrived before the timeout
        # Find response(s) with the highest confidence
        best = None
        ties = []

        for res in results:
            if isinstance(res, dict):
                res = dict2entry(res)
            if res.skill_id in sess.blacklisted_skills:
                LOG.debug(f"ignoring match, skill_id '{res.skill_id}' blacklisted by Session '{sess.session_id}'")
                continue
            if not best or res.match_confidence > best.match_confidence:
                best = res
                ties = [best]
            elif res.match_confidence == best.match_confidence:
                ties.append(res)

        if ties:
            # deterministic tie-break: keep the first candidate encountered
            # (previously used random.choice, which made results non-deterministic)
            selected = ties[0]
            # TODO: Ask user to pick between ties or do it automagically
        else:
            selected = best
        if selected:
            LOG.info(f"OVOSCommonPlay selected: {selected.skill_id} - {selected.match_confidence}")
            LOG.debug(str(selected))
        else:
            LOG.error("No valid OCP matches")
        return selected

    ##################
    # Legacy Audio subsystem API
    def legacy_play(self, results: NormalizedResultsList, phrase="",
                    message: Optional[Message] = None):
        player = self.get_player(message)
        player.media_state = MediaState.LOADING_MEDIA
        playing = False
        for idx, r in enumerate(results):
            real_uri = None
            if not (r.playback == PlaybackType.AUDIO or r.media_type in OCPQuery.cast2audio):
                # we need to filter video results
                continue
            if isinstance(r, Playlist):
                # get internal entries from the playlist
                real_uri = [e.uri for e in r.entries]
            elif isinstance(r, MediaEntry):
                real_uri = r.uri
            elif isinstance(r, PluginStream):
                # for legacy audio service we need to do stream extraction here
                LOG.debug(f"extracting uri: {r.stream}")
                # TODO - apparently it can hang here forever ???
                # happens with https://www.cbc.ca/podcasting/includes/hourlynews.xml from news skill
                try:
                    real_uri = r.extract_uri(video=False)
                except Exception as e:
                    LOG.exception(f"extraction failed: {r}")
            if not real_uri:
                continue
            if not playing:
                playing = True
                self.legacy_api.play(real_uri, utterance=phrase, source_message=message)
                player.player_state = PlayerState.PLAYING
                player.skill_id = r.skill_id
                self.update_player_proxy(player)
            else:
                self.legacy_api.queue(real_uri, source_message=message)

    def _handle_legacy_audio_stop(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available:
            player.player_state = PlayerState.STOPPED
            player.media_state = MediaState.NO_MEDIA
            player.skill_id = None
            self.update_player_proxy(player)

    def _handle_legacy_audio_pause(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available and player.player_state == PlayerState.PLAYING:
            player.player_state = PlayerState.PAUSED
            player.media_state = MediaState.LOADED_MEDIA
            player = self._update_player_skill_id(player, message)
            self.update_player_proxy(player)

    def _handle_legacy_audio_resume(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available and player.player_state == PlayerState.PAUSED:
            player.player_state = PlayerState.PLAYING
            player.media_state = MediaState.LOADED_MEDIA
            player = self._update_player_skill_id(player, message)
            self.update_player_proxy(player)

    def _handle_legacy_audio_start(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available:
            player.player_state = PlayerState.PLAYING
            player.media_state = MediaState.LOADED_MEDIA
            player = self._update_player_skill_id(player, message)
            self.update_player_proxy(player)

    def _handle_legacy_audio_end(self, message: Message):
        player = self.get_player(message)
        if not player.ocp_available:
            player.player_state = PlayerState.STOPPED
            player.media_state = MediaState.END_OF_MEDIA
            player.skill_id = None
            self.update_player_proxy(player)

    @classmethod
    def _get_closest_lang(cls, lang: str) -> Optional[str]:
        if cls.intent_matchers:
            return closest_lang(standardize_lang(lang), list(cls.intent_matchers.keys()))
        return None

    def shutdown(self):
        if not _is_initialised(self):
            return
        self.default_shutdown()  # remove events registered via self.add_event

    def default_shutdown(self):
        if not _is_initialised(self):
            return
        super().default_shutdown()

    # deprecated
    @property
    def mycroft_cps(self) -> LegacyCommonPlay:
        log_deprecation("self.mycroft_cps is deprecated, use MycroftCPSLegacyPipeline instead", "2.0.0")
        return LegacyCommonPlay(self.bus)

    @deprecated("match_fallback has been renamed match_low", "2.0.0")
    def match_fallback(self, utterances: List[str], lang: str, message: Message = None) -> Optional[IntentHandlerMatch]:
        return self.match_low(utterances, lang, message)

    @deprecated("match_legacy is deprecated! use MycroftCPSLegacyPipeline class directly instead", "2.0.0")
    def match_legacy(self, utterances: List[str], lang: str, message: Message = None) -> Optional[IntentHandlerMatch]:
        """ match legacy mycroft common play skills  (must import from deprecated mycroft module)
        not recommended, legacy support only

        legacy base class at mycroft/skills/common_play_skill.py marked for removal in ovos-core 0.1.0
        """
        return MycroftCPSLegacyPipeline(self.bus, self.config).match(utterances, lang, message)


class MycroftCPSLegacyPipeline(PipelinePlugin, OVOSAbstractApplication):
    def __init__(self, bus: Optional[Union[MessageBusClient, FakeBus]] = None,
                 config: Optional[Dict] = None):
        OVOSAbstractApplication.__init__(self, bus=bus or FakeBus(),
                                         skill_id=OCP_ID, resources_dir=f"{dirname(__file__)}")
        PipelinePlugin.__init__(self, bus, config)
        self.mycroft_cps = LegacyCommonPlay(self.bus)
        OCPPipelineMatcher.load_intent_files()
        self.add_event("ocp:legacy_cps", self.handle_legacy_cps, is_intent=True)

    ############
    # Legacy Mycroft CommonPlay skills
    def match(self, utterances: List[str], lang: str, message: Message = None) -> Optional[IntentHandlerMatch]:
        """ match legacy mycroft common play skills  (must import from deprecated mycroft module)
        not recommended, legacy support only

        legacy base class at mycroft/skills/common_play_skill.py marked for removal in ovos-core 0.1.0
        """
        if not self.config.get("legacy_cps", True):
            # needs to be explicitly enabled in pipeline config
            return None

        utterance = utterances[0].lower()

        lang = OCPPipelineMatcher._get_closest_lang(lang)
        if lang is None:  # no intents registered for this lang
            return None

        match = OCPPipelineMatcher.intent_matchers[lang].calc_intent(utterance)
        if hasattr(match, "name"):  # padatious
            match = {
                "name": match.name,
                "conf": match.conf,
                "entities": match.matches
            }

        if match["name"] is None:
            return None
        if match["name"] == "play":
            LOG.info(f"Legacy Mycroft CommonPlay match: {match}")
            utterance = match["entities"].pop("query")
            return IntentHandlerMatch(match_type="ocp:legacy_cps",
                                      match_data={"query": utterance,
                                                  "conf": 0.7},
                                      skill_id=OCP_ID,
                                      utterance=utterance)

    def handle_legacy_cps(self, message: Message):
        """intent handler for legacy CPS matches"""
        utt = message.data["query"]
        res = self.mycroft_cps.search(utt, message=message)
        if res:
            best = OCPPipelineMatcher.select_best([r[0] for r in res], message)
            if best:
                callback = [r[1] for r in res if r[0].uri == best.uri][0]
                self.mycroft_cps.skill_play(skill_id=best.skill_id,
                                            callback_data=callback,
                                            phrase=utt,
                                            message=message)
                return
        self.bus.emit(message.forward("mycroft.audio.play_sound",
                                      {"uri": "snd/error.mp3"}))

    def shutdown(self):
        if not _is_initialised(self):
            return
        self.mycroft_cps.shutdown()

    def default_shutdown(self):
        if not _is_initialised(self):
            return
        super().default_shutdown()
