"""Every call that leaves the machine must carry the working CA bundle.

Some Python builds (uv-managed CPython on RHEL/CentOS) point at a
`/etc/ssl/cert.pem` that does not exist, so a bare `urlopen` fails with
CERTIFICATE_VERIFY_FAILED — which is exactly how the vLLM recipe fetch broke
in v2.7.0 while the DockerHub and GitHub calls kept working.
"""

from __future__ import annotations

import ast
import asyncio
import unittest
from unittest.mock import patch

from tui.common import recipes
from tui.common.profile_store import PROJECT_ROOT

_LOOPBACK_PREFIXES = ("http://localhost", "http://127.")


def _targets_loopback(node: ast.AST) -> bool:
    """True when the urlopen argument is a literal loopback URL."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.startswith(_LOOPBACK_PREFIXES)
    if isinstance(node, ast.JoinedStr) and node.values:
        head = node.values[0]
        return (
            isinstance(head, ast.Constant)
            and isinstance(head.value, str)
            and head.value.startswith(_LOOPBACK_PREFIXES)
        )
    return False


def _loopback_requests(tree: ast.AST) -> set[str]:
    """Names bound to a `Request(<loopback url>)` — urlopen(name) is fine then."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if called != "Request" or not node.value.args:
            continue
        if not _targets_loopback(node.value.args[0]):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


class NoBareUrlopenTests(unittest.TestCase):
    def test_external_calls_go_through_open_url(self) -> None:
        offenders: list[str] = []
        for path in sorted((PROJECT_ROOT / "tui").rglob("*.py")):
            if path.name == "ssl_ctx.py":
                continue
            tree = ast.parse(path.read_text())
            local_requests = _loopback_requests(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (isinstance(func, ast.Attribute) and func.attr == "urlopen"):
                    continue
                if node.args and _targets_loopback(node.args[0]):
                    continue
                if (
                    node.args
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in local_requests
                ):
                    continue
                offenders.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}"
                )
        self.assertEqual(
            offenders, [],
            "these call urlopen directly on a non-loopback target — "
            "use tui.common.ssl_ctx.open_url instead: " + ", ".join(offenders),
        )


class OpenUrlTests(unittest.TestCase):
    def test_attaches_the_ca_context(self) -> None:
        from tui.common import ssl_ctx

        with patch.object(ssl_ctx.urllib.request, "urlopen") as urlopen:
            ssl_ctx.open_url("https://example.invalid", timeout=1)
        self.assertIs(
            urlopen.call_args.kwargs["context"], ssl_ctx.get_ssl_context()
        )


class RecipeFetchUsesContextTests(unittest.TestCase):
    def test_recipe_fetch_uses_open_url(self) -> None:
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b"model:\n  model_id: o/r\n"

        with patch.object(recipes, "open_url", return_value=_Resp()) as opener:
            recipe = asyncio.run(recipes.fetch_recipe("o/r"))
        opener.assert_called_once()
        self.assertEqual(recipe.model_id, "o/r")


class HfListingUsesContextTests(unittest.TestCase):
    def test_hf_tree_listing_uses_open_url(self) -> None:
        from tui.backends.llamacpp import backend as lbackend

        class _Resp:
            headers: dict = {}

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'[{"type": "file", "path": "m.gguf"}]'

        with patch.object(lbackend, "open_url", return_value=_Resp()) as opener:
            files = asyncio.run(lbackend.list_hf_repo_files("o/r"))
        opener.assert_called_once()
        self.assertEqual(files[0]["path"], "m.gguf")


if __name__ == "__main__":
    unittest.main()
