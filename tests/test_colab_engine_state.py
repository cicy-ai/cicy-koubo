import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import app


class ColabEngineStateTest(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()
        self.old_gpu_mode = app.GPU_MODE
        app.GPU_MODE = "colab_cli"
        self.tempdir = tempfile.TemporaryDirectory()
        self.session_config = Path(self.tempdir.name) / "sessions.json"
        self.session_config.write_text(json.dumps({"test-session": {"state": "running"}}))
        self.profile = {
            "session": "test-session",
            "session_config": str(self.session_config),
        }

    def tearDown(self):
        app.GPU_MODE = self.old_gpu_mode
        self.tempdir.cleanup()

    def test_remote_ready_always_clears_installing(self):
        output = []
        for key in app.ENGINES:
            output.extend([
                f"__KOUBO_ENGINE_{key}_BEGIN__",
                "__KOUBO_INSTALLED_YES__",
                "version=test",
                "__KOUBO_INSTALLING__",
                # Simulate a stale/racing process result. READY must win.
                "YES",
                f"__KOUBO_ENGINE_{key}_END__",
            ])

        with mock.patch.object(app, "_active_colab_profile", return_value=self.profile), \
                mock.patch.object(app, "_colab_exec", return_value=(0, "\n".join(output))), \
                mock.patch.object(app, "_colab_engine_ready_marker") as ready_marker, \
                mock.patch.object(app, "_colab_engine_mark_ready"), \
                mock.patch.object(app, "_colab_engine_mark_done"), \
                mock.patch.object(app, "_colab_engine_marked_active", return_value=True):
            ready_marker.return_value.is_file.return_value = False
            response = self.client.get("/api/engines?probe=1")

        self.assertEqual(response.status_code, 200)
        for engine in response.get_json()["engines"]:
            self.assertTrue(engine["installed"], engine["key"])
            self.assertFalse(engine["installing"], engine["key"])

    def test_cached_ready_never_reports_installing(self):
        with mock.patch.object(app, "_active_colab_profile", return_value=self.profile), \
                mock.patch.object(app, "_colab_engine_ready_marker") as ready_marker, \
                mock.patch.object(app, "_colab_engine_marked_active", return_value=True):
            ready_marker.return_value.is_file.return_value = True
            response = self.client.get("/api/engines")

        self.assertEqual(response.status_code, 200)
        for engine in response.get_json()["engines"]:
            self.assertTrue(engine["installed"], engine["key"])
            self.assertFalse(engine["installing"], engine["key"])


class CosyVoicePromptIsolationTest(unittest.TestCase):
    def test_user_reference_text_is_not_primary_segment_conditioning(self):
        script = (
            Path(__file__).parents[1]
            / "scripts/provision/cosy/cosyvoice_tts.py"
        ).read_text()
        self.assertIn("iterator = m.inference_cross_lingual(", script)
        self.assertNotIn(
            "iterator = m.inference_zero_shot(",
            script,
            "zero-shot fallback can leak the reference prompt tail into every segment",
        )


if __name__ == "__main__":
    unittest.main()
