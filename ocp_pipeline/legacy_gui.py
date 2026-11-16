"""The "now playing" page when OCP plays through the legacy audio service.

With ovos-media (or the classic OCP player) a player fills OCP's GUI
namespace (``ovos.common_play``) and shows its page. In legacy mode, OCP
hands its tracks to ovos-audio's AudioService instead, and nothing did:
a device with a screen showed nothing while music played. GUI clients
still have the page, ``OVOSSyncPlayer``, with the values the classic
player set.

:class:`LegacyNowPlaying` fills it for the pipeline, which already is the
``ovos.common_play`` application and knows what it started:

- :meth:`track` remembers the metadata (title, artist, image, length) of
  each URI handed to the audio service, since the audio service only gets
  the address back to us in ``mycroft.audio.playing_track``;
- :meth:`playing`, :meth:`paused`, :meth:`resumed`, :meth:`ended` follow
  the audio service, :meth:`position` follows the backend's
  ``ovos.common_play.playback_time``;
- the page's buttons arrive as ``ovos.common_play.<event>`` (the GUI turns
  a page event into "<namespace>.<event>"); :data:`BUTTONS` maps them to the
  audio service's own messages.

All of it applies only while the legacy audio service plays (the caller
checks ``ocp_available``), so it never competes with ovos-media.
"""
from typing import Any, Dict, Optional

#: the page the classic OCP player showed for audio-service playback
PAGE = "OVOSSyncPlayer"

#: GUI page events (``ovos.common_play.<event>``) -> legacy audio service.
#: ``ovos.common_play.stop`` is also what the stop pipeline sends the active
#: ``ovos.common_play`` application when the user says "stop"; stopping the
#: legacy playback then is what that asks for.
BUTTONS = {
    "pause": "mycroft.audio.service.pause",
    "resume": "mycroft.audio.service.resume",
    "stop": "mycroft.audio.service.stop",
    "next": "mycroft.audio.service.next",
    "previous": "mycroft.audio.service.prev",
}


def _image(meta: Dict[str, Any]) -> str:
    """An image a GUI can load: http(s) or a data: URI."""
    for key in ("image", "skill_icon", "bg_image"):
        value = str(meta.get(key) or "")
        if value.startswith(("http://", "https://", "data:")):
            return value
    return ""


def entry_meta(entry: Any) -> Dict[str, Any]:
    """Title, artist, image and length (ms) of an OCP result object."""
    return {
        "title": getattr(entry, "title", "") or "",
        "artist": getattr(entry, "artist", "") or "",
        "image": getattr(entry, "image", "") or "",
        "skill_icon": getattr(entry, "skill_icon", "") or "",
        "length": int(getattr(entry, "length", 0) or 0),
    }


class LegacyNowPlaying:
    """Shows OCP's now playing page for legacy audio-service playback."""

    def __init__(self, gui):
        self.gui = gui
        self._meta: Dict[str, Dict[str, Any]] = {}   # uri -> metadata
        self.showing = False

    def reset(self):
        """A new play request: forget the metadata of the previous one."""
        self._meta = {}

    def track(self, uri: str, meta: Dict[str, Any]):
        """Remember what ``uri`` is; called for each URI given to the
        audio service (played or queued)."""
        if uri:
            self._meta[uri] = meta

    def playing(self, uri: Optional[str]):
        """``mycroft.audio.playing_track``: show the page for ``uri``."""
        meta = self._meta.get(uri or "") or {}
        g = self.gui
        length = int(meta.get("length") or 0)   # ms; 0 for a live stream
        g["playerBackend"] = PAGE
        g["uri"] = uri or ""
        g["title"] = meta.get("title", "")
        g["artist"] = meta.get("artist", "")
        g["image"] = _image(meta)
        g["bg_image"] = g["image"]
        g["duration"] = length
        g["length"] = length
        g["position"] = 0
        g["status"] = "Playing"
        g["canPrev"] = False
        g["canNext"] = len(self._meta) > 1
        g["loopStatus"] = "None"
        g["shuffleStatus"] = False
        g.show_page(PAGE, override_idle=True)
        self.showing = True

    def paused(self):
        if self.showing:
            self.gui["status"] = "Paused"

    def resumed(self):
        if self.showing:
            self.gui["status"] = "Playing"

    def position(self, position_ms: int, length_ms: int = 0):
        """``ovos.common_play.playback_time`` from the audio backend.

        Only for a track with a known length: a live stream has no position
        worth showing (the page counts the time itself), and a GUI cannot
        tell milliseconds from seconds without a length to compare with."""
        if not self.showing:
            return
        if length_ms:
            self.gui["duration"] = self.gui["length"] = int(length_ms)
        if self.gui.get("length"):
            self.gui["position"] = int(position_ms or 0)

    def ended(self):
        """Stopped, or the end of the queue: the page goes away."""
        if not self.showing:
            return
        self.showing = False
        self.gui["status"] = "Stopped"
        self.gui.release()
