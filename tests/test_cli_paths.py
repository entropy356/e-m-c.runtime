"""CLI storage path failures should produce a concise user-facing error."""
import json
import os
from pathlib import Path
import subprocess
import sys

from .helpers import TempDirTestCase


class CliStoragePathTest(TempDirTestCase):
    def test_invalid_database_paths_report_errors_without_tracebacks(self):
        config_path = self.tmp / 'config.json'
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
        for database in (str(self.tmp), 'bad\x00name'):
            for command in ('history', 'turn'):
                with self.subTest(database=database, command=command):
                    config_path.write_text(json.dumps({
                        'schema_version': 1,
                        'database': database,
                        'provider': {'kind': 'offline'},
                        'modules': [],
                    }), encoding='utf-8')
                    args = [sys.executable, '-m', 'emc_runtime', '--config', str(config_path), command]
                    if command == 'turn':
                        args.append('hello')
                    result = subprocess.run(args, env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 1)
                    self.assertTrue(result.stderr.startswith('Error: '), result.stderr)
                    self.assertNotIn('Traceback', result.stderr)
                    self.assertEqual(result.stdout, '')
