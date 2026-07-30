import importlib
import io
import os
import pathlib
import tempfile
import unittest
from unittest import mock


class GpuApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        os.environ["KOUBO_API_STATE_DIR"] = cls.temp.name
        os.environ["CICY_KOUBO_ACCESS_TOKEN"] = "ga_test_secret"
        cls.server = importlib.import_module("gpu_api.server")
        cls.client = cls.server.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def auth(self):
        return {"Authorization": "Bearer ga_test_secret"}

    def test_live_is_public_but_health_is_authenticated(self):
        self.assertEqual(self.client.get("/live").status_code, 200)
        self.assertEqual(self.client.get("/v1/health").status_code, 401)
        self.assertEqual(
            self.client.get("/v1/health", headers=self.auth()).status_code, 200
        )

    def test_bad_token_is_rejected(self):
        response = self.client.get(
            "/v1/health", headers={"Authorization": "Bearer ga_wrong"}
        )
        self.assertEqual(response.status_code, 401)

    def test_health_declares_multilingual_capabilities(self):
        script = pathlib.Path(self.temp.name) / "cosyvoice_tts.py"
        script.write_text("# multilingual test\n", encoding="utf-8")
        probe = mock.Mock(stdout="NVIDIA T4\n")
        with (
            mock.patch.object(self.server, "COSYVOICE_SCRIPT", script),
            mock.patch.object(self.server.subprocess, "run", return_value=probe),
        ):
            payload = self.client.get(
                "/v1/health", headers=self.auth()
            ).get_json()
        self.assertTrue(payload["capabilities_version"].startswith(
            "2026.07.30-multilingual-"
        ))
        self.assertIn("zh-yue", payload["tts_languages"])
        self.assertIn("lo", payload["tts_languages"])
        self.assertRegex(payload["tts_script_sha256"], r"^[0-9a-f]{64}$")

    def test_tts_rejects_unknown_language(self):
        response = self.client.post(
            "/v1/tts-jobs",
            json={"text": "hello", "language": "xx-invalid"},
            headers=self.auth(),
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "unsupported_tts_language")

    def test_tts_fails_closed_when_multilingual_script_is_missing(self):
        missing = pathlib.Path(self.temp.name) / "missing.py"
        with mock.patch.object(self.server, "COSYVOICE_SCRIPT", missing):
            response = self.client.post(
                "/v1/tts-jobs",
                json={
                    "text": "你好",
                    "language": "zh-yue",
                    "reference_url": "https://example.oss-cn-hangzhou.aliyuncs.com/ref.wav",
                    "reference_text": "你好",
                },
                headers=self.auth(),
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.get_json()["error"], "multilingual_tts_script_missing"
        )

    def test_tts_job_reports_real_segment_progress_and_logs(self):
        job = self.server.STORE.create("tts", "cosyvoice")
        job.status = "running"
        job.stage = "synthesizing"
        job.progress = 20
        (job.directory / "cosyvoice.log").write_text(
            "[02:54:12] 模型就绪(sr=24000),耗时 47.3s\n"
            "[02:54:12] 分句结果: 4 段\n"
            "[02:54:46] 段1 合成完成: 7.80s 音频 / 1 次yield / 耗时 34.4s\n"
            "[02:55:10] 段2 合成完成: 5.20s 音频 / 1 次yield / 耗时 23.1s\n",
            encoding="utf-8",
        )
        payload = self.client.get(
            f"/v1/jobs/{job.id}", headers=self.auth()
        ).get_json()
        self.assertEqual(payload["segments_total"], 4)
        self.assertEqual(payload["segments_completed"], 2)
        self.assertEqual(payload["progress"], 60)
        self.assertEqual(payload["stage"], "synthesizing_segment_3_of_4")
        self.assertTrue(any("模型就绪" in line for line in payload["log"]))

    def test_video_job_reports_real_frame_progress_and_logs(self):
        job = self.server.STORE.create("video", "musetalk")
        job.status = "running"
        job.stage = "lipsync"
        job.progress = 25
        (job.directory / "musetalk.log").write_text(
            "TensorFlow optional backend warning\n"
            "60%|██████    | 138/231 [01:02<00:37, 2.49it/s]\n",
            encoding="utf-8",
        )
        payload = self.client.get(
            f"/v1/jobs/{job.id}", headers=self.auth()
        ).get_json()
        self.assertEqual(payload["progress"], 60)
        self.assertEqual(payload["stage"], "lipsync_frame_138_of_231")
        self.assertEqual(payload["frames_completed"], 138)
        self.assertEqual(payload["frames_total"], 231)
        self.assertIn("TensorFlow optional backend warning", payload["log"])

    def test_job_ids_are_validated(self):
        response = self.client.get("/v1/jobs/nope", headers=self.auth())
        self.assertEqual(response.status_code, 400)

    def test_video_requires_both_uploads(self):
        response = self.client.post(
            "/v1/video-jobs",
            data={"engine": "musetalk"},
            headers=self.auth(),
        )
        self.assertEqual(response.status_code, 400)

    def test_compat_job_contract(self):
        job = self.server.STORE.create("video", "musetalk")
        response = self.client.get(f"/api/job/{job.id}", headers=self.auth())
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("stage", payload)
        self.assertIn("result", payload)
        self.assertIn("error", payload)

    def test_compat_create_returns_legacy_job_id(self):
        with mock.patch.object(self.server.WORK_QUEUE, "put"):
            response = self.client.post(
                "/api/generate-video",
                data={
                    "video": (io.BytesIO(b"video"), "base.mp4"),
                    "audio": (io.BytesIO(b"audio"), "voice.wav"),
                },
                headers=self.auth(),
            )
        self.assertEqual(response.status_code, 202)
        self.assertRegex(response.get_json()["job_id"], r"^gpu_[0-9a-f]{32}$")

    def test_result_supports_conditional_download(self):
        job = self.server.STORE.create("video", "musetalk")
        result = job.directory / "result.mp4"
        result.write_bytes(b"0123456789")
        self.server._finish(job, result)
        response = self.client.get(
            f"/v1/jobs/{job.id}/result",
            headers={**self.auth(), "Range": "bytes=2-5"},
        )
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.data, b"2345")
        self.assertIn("X-Checksum-SHA256", response.headers)
        response.close()


if __name__ == "__main__":
    unittest.main()
