from __future__ import annotations

import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner


async def _collect_events(generator) -> list[tuple[str, object]]:
    return [event async for event in generator]


class CredentialBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def test_reference_validators_reject_untrimmed_and_control_input(self) -> None:
        from tui.common.dev_build import (
            image_reference_credential_error,
            repo_url_error,
        )

        for value in (
            " registry.example/org/image:v1",
            "registry.example/org/image:v1 ",
            "registry.example/org/image:v1\nignored",
        ):
            with self.subTest(image=value):
                self.assertTrue(image_reference_credential_error(value))
        for value in (
            "https://git.example/org/repo.git\nignored",
            "https://git.example/org/repo.git\x7fignored",
        ):
            with self.subTest(repo=value):
                self.assertTrue(repo_url_error(value))

    async def test_image_inspectors_reject_credentials_before_docker(self) -> None:
        from tui.backends.llamacpp import backend as llamacpp
        from tui.backends.vllm import backend_inspect as vllm
        from tui.common import docker

        secret = "llmux-r3-inspect-secret"
        image = f"builder:{secret}@registry.example/org/image:v1"
        identity = AsyncMock(side_effect=AssertionError("docker must not run"))
        with patch.object(docker, "image_identity", identity):
            for inspector in (vllm.extract_vllm_params, llamacpp.extract_llama_server_flags):
                with self.subTest(inspector=inspector.__module__):
                    with self.assertRaises(RuntimeError) as caught:
                        await inspector(image)
                    self.assertNotIn(secret, str(caught.exception))
        identity.assert_not_awaited()

    async def test_resolved_default_repo_is_rejected_before_git_or_output(self) -> None:
        from tui.common import dev_build

        secret = "llmux-r3-git-secret"
        repo = f"https://builder:{secret}@git.example/repo.git"
        spec = dev_build.DevBuildSpec(
            backend="synthetic",
            image_prefix="synthetic-dev",
            src_dir=Path("/tmp/llmux-r3-never-created"),
            default_repo_url=repo,
        )
        clone = MagicMock(side_effect=AssertionError("git must not run"))
        with patch.object(dev_build, "clone_or_update", clone):
            events = await _collect_events(dev_build.stream_build(spec, "main"))

        rendered = "\n".join(str(value) for _, value in events)
        self.assertNotIn(secret, rendered)
        self.assertEqual(events[-1], ("rc", 1))
        clone.assert_not_called()

    async def test_resolved_image_prefix_is_rejected_before_git(self) -> None:
        from tui.common import dev_build

        spec = dev_build.DevBuildSpec(
            backend="synthetic",
            image_prefix="synthetic-dev?token=secret",
            src_dir=Path("/tmp/llmux-r3-never-created"),
            default_repo_url="https://git.example/repo.git",
        )
        clone = MagicMock(side_effect=AssertionError("git must not run"))
        with patch.object(dev_build, "clone_or_update", clone):
            events = await _collect_events(dev_build.stream_build(spec, "main"))

        self.assertEqual(events[-1], ("rc", 1))
        clone.assert_not_called()

    async def test_clone_boundary_revalidates_repo(self) -> None:
        from tui.common import dev_build

        secret = "llmux-r3-clone-secret"
        repo = f"https://builder:{secret}@git.example/repo.git"
        spec = dev_build.DevBuildSpec(
            backend="synthetic",
            image_prefix="synthetic-dev",
            src_dir=Path("/tmp/llmux-r3-never-created"),
            default_repo_url="https://git.example/repo.git",
        )
        transport = MagicMock(side_effect=AssertionError("transport must not be built"))
        with patch.object(dev_build, "_git_transport", transport):
            events = await _collect_events(dev_build.clone_or_update(spec, repo, "main"))

        rendered = "\n".join(str(value) for _, value in events)
        self.assertNotIn(secret, rendered)
        self.assertEqual(events[-1], ("rc", 1))
        transport.assert_not_called()


class HfFileBoundaryTests(unittest.TestCase):
    def test_hf_file_requires_normalized_relative_posix_path(self) -> None:
        from tui.common.prepare import hf_file_error

        self.assertEqual(hf_file_error("quant/model.gguf"), "")
        for value in (
            "",
            "/tmp/outside.gguf",
            "../outside.gguf",
            "quant/../outside.gguf",
            "./model.gguf",
            "quant//model.gguf",
            "model.gguf\nignored",
        ):
            with self.subTest(value=value):
                self.assertTrue(hf_file_error(value))

    def test_prepare_rejects_absolute_file_before_cache_probe(self) -> None:
        from tui.common import prepare

        with tempfile.TemporaryDirectory() as tmpdir:
            outside = Path(tmpdir) / "outside.gguf"
            outside.write_bytes(b"secret")
            with self.assertRaisesRegex(ValueError, "relative"):
                prepare.gguf_in_cache(tmpdir, "org/repo", str(outside))


class HfRepoBoundaryTests(unittest.TestCase):
    def test_profile_rejects_noncanonical_hf_repo(self) -> None:
        from tui.common import profile_store

        for value in (
            "/org/repo",
            "org/repo/extra",
            "../repo",
            "org/..",
            "org/repo\nignored",
        ):
            with self.subTest(value=value):
                profile = profile_store.StoredProfile(
                    name="audit",
                    backend="llamacpp",
                    port=8080,
                    hf_repo=value,
                )
                with self.assertRaisesRegex(ValueError, "HF repo"):
                    profile_store._validate_profile(profile)

    def test_repo_cache_symlink_outside_configured_root_is_rejected(self) -> None:
        from tui.backends.llamacpp import backend
        from tui.common import prepare

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache = root / "cache"
            outside_repo = root / "outside-repo"
            snapshot = outside_repo / "snapshots" / "revision"
            refs = outside_repo / "refs"
            snapshot.mkdir(parents=True)
            refs.mkdir()
            (refs / "main").write_text("revision")
            (snapshot / "model.gguf").write_bytes(b"outside")
            hub = cache / "hub"
            hub.mkdir(parents=True)
            (hub / "models--org--repo").symlink_to(
                outside_repo, target_is_directory=True
            )

            with self.assertRaisesRegex(RuntimeError, "outside"):
                prepare.gguf_in_cache(str(cache), "org/repo", "model.gguf")
            with patch.object(backend, "_get_hf_cache_dir", return_value=cache):
                with self.assertRaisesRegex(RuntimeError, "outside"):
                    backend.find_cached_gguf("org/repo", "model.gguf")


class GitBranchBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_branch_is_rejected_before_git(self) -> None:
        from tui.common import dev_build

        spec = dev_build.DevBuildSpec(
            backend="synthetic",
            image_prefix="synthetic-dev",
            src_dir=Path("/tmp/llmux-r3-never-created"),
            default_repo_url="https://git.example/repo.git",
        )
        transport = MagicMock(side_effect=AssertionError("git must not run"))
        with patch.object(dev_build, "_git_transport", transport):
            for branch in (
                "--detach",
                "feature/../main",
                "feature/.hidden",
                "feature/name.lock",
                "feature/name@{1}",
                "feature/name?query",
                "feature/name\nignored",
            ):
                with self.subTest(branch=branch):
                    events = await _collect_events(
                        dev_build.clone_or_update(
                            spec,
                            "https://git.example/repo.git",
                            branch,
                        )
                    )
                    self.assertEqual(events[-1], ("rc", 1))
        transport.assert_not_called()

    def test_valid_branch_names_pass_shared_boundary(self) -> None:
        from tui.common.dev_build import git_branch_error

        for branch in ("main", "feature/cache-boundary", "release/v1.2.3"):
            with self.subTest(branch=branch):
                self.assertEqual(git_branch_error(branch), "")


class HfRevisionAndSymlinkTests(unittest.TestCase):
    def _repo(self, root: Path, revision: str = "bbb-current") -> Path:
        repo = root / "hub" / "models--org--repo"
        (repo / "refs").mkdir(parents=True)
        (repo / "refs" / "main").write_text(revision)
        return repo

    def test_cache_hit_uses_refs_main_instead_of_sorted_snapshot(self) -> None:
        from tui.backends.llamacpp import backend

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo = self._repo(root)
            stale = repo / "snapshots" / "aaa-stale"
            current = repo / "snapshots" / "bbb-current"
            stale.mkdir(parents=True)
            current.mkdir(parents=True)
            (stale / "model.gguf").write_bytes(b"stale")

            with patch.object(backend, "_get_hf_cache_dir", return_value=root):
                self.assertIsNone(backend.find_cached_gguf("org/repo", "model.gguf"))
                expected = current / "model.gguf"
                expected.write_bytes(b"current")
                self.assertEqual(
                    backend.find_cached_gguf("org/repo", "model.gguf"), expected
                )

    def test_repo_blob_symlink_is_valid_and_inventory_has_revision(self) -> None:
        from tui.backends.llamacpp import backend

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo = self._repo(root)
            snapshot = repo / "snapshots" / "bbb-current"
            blob = repo / "blobs" / "sha256"
            snapshot.mkdir(parents=True)
            blob.parent.mkdir()
            blob.write_bytes(b"trusted")
            entry = snapshot / "model.gguf"
            entry.symlink_to("../../blobs/sha256")

            with patch.object(backend, "_get_hf_cache_dir", return_value=root):
                self.assertEqual(
                    backend.find_cached_gguf("org/repo", "model.gguf"), entry
                )
                inventory = backend.list_cached_gguf()

        self.assertEqual(len(inventory), 1)
        self.assertEqual(inventory[0]["revision"], "bbb-current")
        self.assertEqual(inventory[0]["size_bytes"], len(b"trusted"))

    def test_cache_outside_symlink_is_an_explicit_error(self) -> None:
        from tui.backends.llamacpp import backend

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo = self._repo(root)
            snapshot = repo / "snapshots" / "bbb-current"
            snapshot.mkdir(parents=True)
            outside = root / "outside.gguf"
            outside.write_bytes(b"outside")
            (snapshot / "model.gguf").symlink_to(outside)

            with patch.object(backend, "_get_hf_cache_dir", return_value=root):
                with self.assertRaisesRegex(RuntimeError, "outside"):
                    backend.find_cached_gguf("org/repo", "model.gguf")
                with self.assertRaisesRegex(RuntimeError, "outside"):
                    backend.list_cached_gguf()


class OriginBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_hf_listing_rejects_noncanonical_repo_before_request(self) -> None:
        from tui.backends.llamacpp import backend

        opener = MagicMock(side_effect=AssertionError("network must not run"))
        with patch.object(backend, "open_url", opener):
            with self.assertRaisesRegex(backend.HfListingUnavailable, "canonical"):
                await backend.list_hf_repo_files("../repo")
        opener.assert_not_called()

    async def test_hf_off_origin_next_is_rejected_before_second_request(self) -> None:
        from tui.backends.llamacpp import backend

        class Response:
            headers = {
                "Link": '<http://127.0.0.1:2375/sentinel>; rel="next"'
            }

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self) -> bytes:
                return b'[{"type":"file","path":"model.gguf"}]'

        opener = MagicMock(return_value=Response())
        with patch.object(backend, "open_url", opener):
            with self.assertRaisesRegex(backend.HfListingUnavailable, "off-origin"):
                await backend.list_hf_repo_files("org/repo")
        self.assertEqual(opener.call_count, 1)

    async def test_dockerhub_off_origin_next_is_rejected(self) -> None:
        from tui.backends.vllm import backend_inspect

        fetch = AsyncMock(
            return_value={
                "results": [],
                "next": "http://127.0.0.1:2375/sentinel",
            }
        )
        with patch.object(backend_inspect, "_fetch_json_url", fetch):
            with self.assertRaisesRegex(RuntimeError, "off-origin"):
                await backend_inspect.get_dockerhub_release_version()
        fetch.assert_awaited_once()

    async def test_dockerhub_collects_all_pages_before_selecting_stable(self) -> None:
        from tui.backends.vllm import backend_inspect

        first = (
            "https://hub.docker.com/v2/repositories/vllm/vllm-openai/"
            "tags?page_size=100"
        )
        second = f"{first}&page=2"
        pages = {
            first: {
                "results": [{"name": "v0.20.0"}],
                "next": second,
            },
            second: {
                "results": [{"name": "v0.21.0"}],
                "next": None,
            },
        }

        async def fetch(url: str, **_kwargs):
            return pages[url]

        with patch.object(backend_inspect, "_fetch_json_url", side_effect=fetch) as call:
            version = await backend_inspect.get_dockerhub_release_version()

        self.assertEqual(version, "v0.21.0")
        self.assertEqual(call.await_count, 2)


class RedirectBoundaryTests(unittest.TestCase):
    def test_off_origin_redirect_is_not_followed(self) -> None:
        from tui.common.ssl_ctx import _SameOriginRedirectHandler

        handler = _SameOriginRedirectHandler()
        request = urllib.request.Request(
            "https://hf.example/private",
            headers={"Authorization": "Bearer secret"},
        )
        redirected = handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "http://127.0.0.1:2375/sentinel",
        )
        self.assertIsNone(redirected)


class ImageInventoryBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_rows_fail_all_owned_inventory_paths(self) -> None:
        from tui.backends.llamacpp import backend as llamacpp
        from tui.backends.vllm import backend_inspect as vllm
        from tui.common import dev_build

        spec = dev_build.DevBuildSpec(
            backend="synthetic",
            image_prefix="synthetic-dev",
            src_dir=Path("/tmp/llmux-r3-never-created"),
            default_repo_url="https://git.example/repo.git",
        )
        with patch.object(vllm, "run_command", AsyncMock(return_value=(0, "malformed\n"))):
            with self.assertRaises(RuntimeError):
                await vllm.get_docker_images()
        with patch.object(
            llamacpp, "run_command", AsyncMock(return_value=(0, "malformed\n"))
        ):
            with self.assertRaises(RuntimeError):
                await llamacpp.get_docker_images()
        with patch.object(dev_build, "_run", AsyncMock(return_value=(0, "malformed\n"))):
            with self.assertRaises(RuntimeError):
                await dev_build.list_local_dev_images(spec)


class ImagePullBoundaryTests(unittest.TestCase):
    def test_latest_is_rejected_before_pull(self) -> None:
        from tui.cli import image

        pull = AsyncMock(return_value=(0, []))
        with patch.object(image.system_operations, "pull_image", pull):
            result = CliRunner().invoke(image.app, ["pull", "latest"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("latest", result.output)
        pull.assert_not_awaited()


class LlamacppConfigLoaderBoundaryTests(unittest.TestCase):
    def test_duplicate_mapping_keys_are_rejected(self) -> None:
        from tui.backends.llamacpp import backend

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            (config_dir / "duplicate.yaml").write_text(
                "ctx-size: 1024\nctx-size: 2048\n"
            )
            with patch.object(backend, "CONFIG_DIR", config_dir):
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    backend.load_config("duplicate")


if __name__ == "__main__":
    unittest.main()
