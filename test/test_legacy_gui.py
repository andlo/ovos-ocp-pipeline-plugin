"""The now playing page for playback through the legacy audio service."""
import unittest

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus
from ovos_utils.ocp import MediaEntry, MediaType, PlaybackType

from ocp_pipeline.legacy_gui import BUTTONS, PAGE, LegacyNowPlaying, entry_meta
from ocp_pipeline.opm import OCPPipelineMatcher


class FakeGUI(dict):
    """The part of a skill GUI the page uses."""

    def __init__(self):
        super().__init__()
        self.pages = []
        self.released = False

    def show_page(self, name, override_idle=None, **kwargs):
        self.pages.append(name)

    def release(self):
        self.released = True
        self.clear()


STATION = MediaEntry(title="DR P1", artist="radio-browser", uri="https://live.example/p1.mp3",
                     image="https://example/p1.png", media_type=MediaType.RADIO,
                     playback=PlaybackType.AUDIO, length=0)


class TestLegacyNowPlaying(unittest.TestCase):

    def test_page_shows_what_the_audio_service_plays(self):
        gui = FakeGUI()
        page = LegacyNowPlaying(gui)
        page.track(STATION.uri, entry_meta(STATION))
        page.playing(STATION.uri)
        self.assertEqual(gui.pages, [PAGE])
        self.assertEqual(gui["title"], "DR P1")
        self.assertEqual(gui["artist"], "radio-browser")
        self.assertEqual(gui["image"], "https://example/p1.png")
        self.assertEqual(gui["status"], "Playing")
        self.assertEqual(gui["duration"], 0)          # live
        self.assertEqual(gui["playerBackend"], PAGE)

    def test_status_position_and_end(self):
        gui = FakeGUI()
        page = LegacyNowPlaying(gui)
        page.track("https://x/a.mp3", {"title": "A", "length": 180000})
        page.playing("https://x/a.mp3")
        page.position(5000, 180000)
        self.assertEqual(gui["position"], 5000)
        page.paused()
        self.assertEqual(gui["status"], "Paused")
        page.resumed()
        self.assertEqual(gui["status"], "Playing")
        page.ended()
        self.assertTrue(gui.released)
        self.assertFalse(page.showing)

    def test_unknown_uri_still_shows_the_page(self):
        gui = FakeGUI()
        page = LegacyNowPlaying(gui)
        page.playing("https://x/unknown.mp3")
        self.assertEqual(gui["uri"], "https://x/unknown.mp3")
        self.assertEqual(gui["title"], "")

    def test_nothing_before_playback(self):
        gui = FakeGUI()
        page = LegacyNowPlaying(gui)
        page.paused()
        page.position(1000)
        page.ended()
        self.assertEqual(gui, {})
        self.assertFalse(gui.released)


class TestPipelineLegacyGUI(unittest.TestCase):
    """The pipeline drives the page from the legacy audio service's messages."""

    def setUp(self):
        self.bus = FakeBus()
        self.sent = []
        self.bus.on("message", lambda m: self.sent.append(
            Message.deserialize(m) if isinstance(m, str) else m))
        self.ocp = OCPPipelineMatcher(bus=self.bus, config={"legacy": True})
        self.ocp.legacy_gui.gui = self.gui = FakeGUI()

    def tearDown(self):
        self.ocp.shutdown()

    def types(self):
        return [m.msg_type for m in self.sent]

    def test_play_shows_and_buttons_reach_the_audio_service(self):
        self.ocp.legacy_play([STATION], "dr p1", message=Message("ocp:play"))
        self.assertIn("mycroft.audio.service.play", self.types())
        self.bus.emit(Message("mycroft.audio.playing_track", {"track": STATION.uri}))
        self.assertEqual(self.gui["title"], "DR P1")
        self.assertEqual(self.gui.pages, [PAGE])

        # stop last: it closes the page, after which buttons are left alone
        for event in ("pause", "resume", "next", "previous", "stop"):
            self.bus.emit(Message(f"ovos.common_play.{event}"))
            self.assertIn(BUTTONS[event], self.types(), event)
        self.assertTrue(self.gui.released)

    def test_seek(self):
        self.ocp.legacy_play([STATION], "dr p1", message=Message("ocp:play"))
        self.bus.emit(Message("mycroft.audio.playing_track", {"track": STATION.uri}))
        self.bus.emit(Message("ovos.common_play.seek", {"seekValue": 4000}))
        seek = [m for m in self.sent if m.msg_type == "mycroft.audio.service.set_track_position"]
        self.assertEqual(seek[-1].data["position"], 4000)

    def test_buttons_are_left_alone_when_the_page_is_not_shown(self):
        self.bus.emit(Message("ovos.common_play.pause"))
        self.assertNotIn("mycroft.audio.service.pause", self.types())

    def test_end_of_queue_closes_the_page(self):
        self.ocp.legacy_play([STATION], "dr p1", message=Message("ocp:play"))
        self.bus.emit(Message("mycroft.audio.playing_track", {"track": STATION.uri}))
        # a live stream: the page counts the time itself
        self.bus.emit(Message("ovos.common_play.playback_time", {"position": 3000, "length": 0}))
        self.assertEqual(self.gui["position"], 0)
        self.bus.emit(Message("mycroft.audio.queue_end"))
        self.assertTrue(self.gui.released)


if __name__ == "__main__":
    unittest.main()
