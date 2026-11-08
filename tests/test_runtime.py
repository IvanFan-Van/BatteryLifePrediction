"""Environment hygiene for queued training."""

import os
import unittest
from unittest.mock import patch

from src.utils.runtime import clean_environment


class RuntimeTests(unittest.TestCase):
    def test_task_environment_excludes_credentials(self):
        with patch.dict(os.environ,{'EXAMPLE_API_KEY':'dummy-value','HF_TOKEN':'dummy-value'}):
            env=clean_environment()
        self.assertNotIn('EXAMPLE_API_KEY',env)
        self.assertNotIn('HF_TOKEN',env)
        self.assertEqual(env['WANDB_MODE'],'offline')
