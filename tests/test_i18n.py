import os
import unittest
from unittest.mock import patch

from tui.common import i18n


class I18nTests(unittest.TestCase):
    def _env(self, **overrides):
        # Start from a clean slate so the host's locale doesn't leak in.
        base = {k: "" for k in ("LLMUX_LANG", "LC_ALL", "LC_MESSAGES", "LANG")}
        base.update(overrides)
        return patch.dict(os.environ, base, clear=False)

    def test_explicit_llmux_lang_wins(self):
        with self._env(LLMUX_LANG="ko", LANG="en_US.UTF-8"):
            self.assertEqual(i18n.lang(), "ko")
            self.assertEqual(i18n.t("hi", "안녕"), "안녕")
        with self._env(LLMUX_LANG="en", LANG="ko_KR.UTF-8"):
            self.assertEqual(i18n.lang(), "en")
            self.assertEqual(i18n.t("hi", "안녕"), "hi")

    def test_falls_back_to_locale(self):
        with self._env(LANG="ko_KR.UTF-8"):
            self.assertEqual(i18n.lang(), "ko")
        with self._env(LC_ALL="ko_KR.UTF-8", LANG="en_US.UTF-8"):
            self.assertEqual(i18n.lang(), "ko")  # LC_ALL outranks LANG

    def test_defaults_to_english(self):
        with self._env(LANG="en_US.UTF-8"):
            self.assertEqual(i18n.lang(), "en")
        with self._env():  # nothing set
            self.assertEqual(i18n.lang(), "en")

    def test_invalid_llmux_lang_ignored(self):
        with self._env(LLMUX_LANG="fr", LANG="ko_KR.UTF-8"):
            self.assertEqual(i18n.lang(), "ko")  # bad override → locale


if __name__ == "__main__":
    unittest.main()
