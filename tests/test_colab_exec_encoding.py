import pathlib
import subprocess
import sys
import unittest
from unittest import mock

import src.app as app


class ColabExecEncodingTests(unittest.TestCase):
    def test_invalid_utf8_from_remote_command_is_replaced_not_failed(self):
        real_run = subprocess.run

        def execute_uploaded_script(args, **_kwargs):
            script = pathlib.Path(args[args.index("-f") + 1])
            return real_run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )

        profile = {"id": "test", "session": "test-session", "auth": "oauth2"}
        command = (
            f"{sys.executable} -c "
            "\"import os; os.write(1, b'prefix\\\\xffsuffix')\""
        )
        with mock.patch.object(app, "_colab_resolve_session", return_value="test-session"), \
                mock.patch.object(app, "_colab_base_args", return_value=["fake-colab"]), \
                mock.patch.object(app.subprocess, "run", side_effect=execute_uploaded_script):
            rc, output = app._colab_exec(command, timeout=10, profile=profile)

        self.assertEqual(rc, 0)
        self.assertIn("prefix", output)
        self.assertIn("\ufffd", output)
        self.assertIn("suffix", output)


if __name__ == "__main__":
    unittest.main()
