"""A bare "stop" while OCP plays (the stop pipeline, OVOS-STOP-1).

OCP answers the stop ping with can_handle while it plays or is paused, and
stops the playback when the stop pipeline picks it. A result from an
in-process MediaProvider makes OCP the active skill, not the provider,
whose name is no skill on the bus.
"""
import unittest
from unittest.mock import MagicMock

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session
from ovos_utils.ocp import PlayerState

from ocp_pipeline.opm import OCPPipelineMatcher, OCPPlayerProxy, OCP_ID


class TestStop(unittest.TestCase):
    def setUp(self):
        self.ocp = OCPPipelineMatcher(config={"legacy": True})
        self.sess = Session("stop-test")
        self.msg = Message("ovos.stop.ping", context={"session": self.sess.serialize()})

    def tearDown(self):
        self.ocp.shutdown()

    def _player(self, state):
        player = OCPPlayerProxy(available_extractors=[], ocp_available=False,
                                session_id=self.sess.session_id,
                                player_state=state)
        self.ocp.ocp_sessions[self.sess.session_id] = player
        return player

    def test_can_stop_follows_the_player(self):
        self.assertFalse(self.ocp.can_stop(self.msg))       # nothing played yet
        self._player(PlayerState.PLAYING)
        self.assertTrue(self.ocp.can_stop(self.msg))
        self._player(PlayerState.PAUSED)
        self.assertTrue(self.ocp.can_stop(self.msg))
        self._player(PlayerState.STOPPED)
        self.assertFalse(self.ocp.can_stop(self.msg))

    def test_stop_session_stops_legacy_playback(self):
        player = self._player(PlayerState.PLAYING)
        self.ocp.legacy_api = MagicMock()
        self.assertTrue(self.ocp.stop_session(self.sess))
        self.ocp.legacy_api.stop.assert_called_once()
        self.assertEqual(self.ocp.ocp_sessions[self.sess.session_id].player_state,
                         PlayerState.STOPPED)
        # and nothing to stop the second time
        self.assertFalse(self.ocp.stop_session(self.sess))

    def _play(self, skill_id):
        """handle_play_intent with one result from ``skill_id``; returns the
        skill ids the Session was told to activate."""
        best = MagicMock(skill_id=skill_id, media_type=0, as_dict={})
        self.ocp._search = MagicMock(return_value=[best])
        self.ocp.select_best = MagicMock(return_value=best)
        self.ocp.speak_dialog = MagicMock()
        self.ocp.legacy_play = MagicMock()
        self.ocp.ocp_api = MagicMock()
        self._player(PlayerState.STOPPED)
        activated = []
        orig = Session.activate_skill
        Session.activate_skill = lambda s, sid: activated.append(sid)
        try:
            self.ocp.handle_play_intent(Message(
                "ocp:play", {"lang": "da-DK", "query": "dr p1", "media_type": 0},
                {"session": self.sess.serialize()}))
        finally:
            Session.activate_skill = orig
        return activated

    def test_provider_result_activates_ocp(self):
        """The provider's name never lands in the active skills."""
        self.ocp.media_providers = {"pyradios": MagicMock()}
        self.assertEqual(self._play("pyradios"), [OCP_ID])

    def test_skill_result_activates_the_skill(self):
        """A result from an OCP skill on the bus still activates that skill."""
        self.ocp.skill_aliases = {"ovos-skill-somafm.openvoiceos": ["somafm"]}
        self.assertEqual(self._play("ovos-skill-somafm.openvoiceos"),
                         ["ovos-skill-somafm.openvoiceos"])


if __name__ == "__main__":
    unittest.main()
