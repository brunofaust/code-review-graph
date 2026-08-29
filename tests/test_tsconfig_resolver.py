"""Tests for the TsconfigResolver class."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from code_review_graph.parser import CodeParser
from code_review_graph.tsconfig_resolver import TsconfigResolver

FIXTURES = Path(__file__).parent / "fixtures"


def _write_config(root: Path, name: str, paths: dict, base_url: str = ".") -> None:
    (root / name).write_text(
        json.dumps({"compilerOptions": {"baseUrl": base_url, "paths": paths}}),
        encoding="utf-8",
    )


def _guard_external_os_access(repo: Path, monkeypatch):
    """Fail if resolver path handling asks the OS about anything outside repo."""
    original_lstat = os.lstat
    original_readlink = os.readlink
    original_stat = os.stat
    readlink_calls: list[Path] = []

    def checked_path(raw_path) -> Path:
        path = Path(os.path.abspath(os.fspath(raw_path)))
        assert path == repo or repo in path.parents, f"external filesystem probe: {path}"
        return path

    def guarded_lstat(path, *args, **kwargs):
        checked_path(path)
        return original_lstat(path, *args, **kwargs)

    def guarded_readlink(path, *args, **kwargs):
        readlink_calls.append(checked_path(path))
        return original_readlink(path, *args, **kwargs)

    def guarded_stat(path, *args, **kwargs):
        checked_path(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", guarded_lstat)
    monkeypatch.setattr(os, "readlink", guarded_readlink)
    monkeypatch.setattr(os, "stat", guarded_stat)
    return readlink_calls


def _swap_after_path_precheck(target: Path, replacement: Path, monkeypatch, lstat_count: int):
    """Replace target after legacy lstat or descriptor fstat verification."""
    original_lstat = os.lstat
    original_fstat = os.fstat
    target_identity = original_lstat(target)
    state = {"lstats": 0, "swapped": False}

    def swap() -> None:
        target.unlink()
        target.symlink_to(replacement)
        state["swapped"] = True

    def racing_lstat(path, *args, **kwargs):
        result = original_lstat(path, *args, **kwargs)
        if Path(path) == target and not state["swapped"]:
            state["lstats"] += 1
            if state["lstats"] == lstat_count:
                swap()
        return result

    def racing_fstat(fd):
        result = original_fstat(fd)
        if (
            not state["swapped"]
            and result.st_dev == target_identity.st_dev
            and result.st_ino == target_identity.st_ino
        ):
            swap()
        return result

    monkeypatch.setattr(os, "lstat", racing_lstat)
    monkeypatch.setattr(os, "fstat", racing_fstat)
    return state


class TestTsconfigResolver:
    def setup_method(self):
        self.resolver = TsconfigResolver(FIXTURES)

    def test_strip_jsonc_comments(self):
        text = '{\n  // comment\n  "key": "value" /* block */\n}'
        result = self.resolver._strip_jsonc_comments(text)
        assert "//" not in result
        assert "/*" not in result

    def test_strip_trailing_commas(self):
        text = '{"a": 1, "b": 2,}'
        result = self.resolver._strip_jsonc_comments(text)
        assert ",}" not in result

    def test_resolve_alias(self):
        importer = str(FIXTURES / "alias_importer.ts")
        result = self.resolver.resolve_alias("@/lib/utils", importer)
        assert result is not None
        assert result.endswith("utils.ts")

    def test_resolve_alias_nonexistent_returns_none(self):
        importer = str(FIXTURES / "alias_importer.ts")
        result = self.resolver.resolve_alias("@/nonexistent/module", importer)
        assert result is None

    def test_resolve_npm_package_returns_none(self):
        importer = str(FIXTURES / "alias_importer.ts")
        result = self.resolver.resolve_alias("react", importer)
        assert result is None

    def test_no_tsconfig_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = str(Path(tmp_dir) / "file.ts")
            result = self.resolver.resolve_alias("@/foo", file_path)
        assert result is None

    def test_caching(self):
        importer = str(FIXTURES / "alias_importer.ts")
        self.resolver.resolve_alias("@/lib/utils", importer)
        cache_size_after_first = len(self.resolver._cache)
        assert cache_size_after_first >= 1
        self.resolver.resolve_alias("@/lib/utils", importer)
        assert len(self.resolver._cache) == cache_size_after_first


class TestJsconfigResolution:
    """Regression tests for issue #776: jsconfig.json path aliases."""

    def test_jsconfig_only_project_resolves_alias(self):
        """A plain-JS project declaring aliases only in jsconfig.json resolves them."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            _write_config(root, "jsconfig.json", {"@/*": ["src/*"]})
            target = root / "src" / "composables" / "useThing.js"
            target.parent.mkdir(parents=True)
            target.write_text("export function useThing() {}\n", encoding="utf-8")
            importer = root / "src" / "App.vue"
            importer.write_text("import '@/composables/useThing'\n", encoding="utf-8")

            result = TsconfigResolver(root).resolve_alias("@/composables/useThing", str(importer))
            assert result is not None
            assert Path(result) == target.resolve()

    def test_tsconfig_wins_over_jsconfig_in_same_dir(self):
        """When both configs exist in a directory, tsconfig.json takes precedence."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            _write_config(root, "tsconfig.json", {"@/*": ["ts_src/*"]})
            _write_config(root, "jsconfig.json", {"@/*": ["js_src/*"]})
            ts_target = root / "ts_src" / "mod.ts"
            ts_target.parent.mkdir(parents=True)
            ts_target.write_text("export const x = 1\n", encoding="utf-8")
            js_target = root / "js_src" / "mod.js"
            js_target.parent.mkdir(parents=True)
            js_target.write_text("export const x = 1\n", encoding="utf-8")
            importer = root / "main.ts"
            importer.write_text("import { x } from '@/mod'\n", encoding="utf-8")

            result = TsconfigResolver(root).resolve_alias("@/mod", str(importer))
            assert result is not None
            assert Path(result) == ts_target.resolve()

    def test_jsconfig_with_jsonc_comments_and_extends(self):
        """jsconfig files support JSONC comments and relative extends chains."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "jsconfig.base.json").write_text(
                '{\n'
                '  // shared aliases\n'
                '  "compilerOptions": {\n'
                '    "baseUrl": ".",\n'
                '    "paths": {"@/*": ["src/*"],}\n'
                '  }\n'
                '}\n',
                encoding="utf-8",
            )
            (root / "jsconfig.json").write_text(
                '{"extends": "./jsconfig.base.json", "compilerOptions": {}}\n',
                encoding="utf-8",
            )
            target = root / "src" / "util.js"
            target.parent.mkdir(parents=True)
            target.write_text("export const u = 1\n", encoding="utf-8")
            importer = root / "src" / "app.js"
            importer.write_text("import { u } from '@/util'\n", encoding="utf-8")

            result = TsconfigResolver(root).resolve_alias("@/util", str(importer))
            assert result is not None
            assert Path(result) == target.resolve()


class TestRepositoryContainment:
    @pytest.mark.parametrize("config_name", ["tsconfig.json", "jsconfig.json"])
    def test_config_swap_after_precheck_reads_verified_inode(
        self,
        tmp_path,
        monkeypatch,
        config_name,
    ):
        repo = tmp_path / "repo"
        outside = tmp_path / "outside"
        repo.mkdir()
        outside.mkdir()
        importer = repo / "app.ts"
        importer.write_text("import '@/secret'\n", encoding="utf-8")
        _write_config(repo, config_name, {"@/*": ["safe/*"]})
        external_config = outside / config_name
        _write_config(outside, config_name, {"@/*": ["evil/*"]})
        safe_target = repo / "safe" / "secret.ts"
        evil_target = repo / "evil" / "secret.ts"
        safe_target.parent.mkdir()
        evil_target.parent.mkdir()
        safe_target.write_text("export const safe = 1\n", encoding="utf-8")
        evil_target.write_text("export const evil = 1\n", encoding="utf-8")

        with monkeypatch.context() as context:
            state = _swap_after_path_precheck(
                repo / config_name,
                external_config,
                context,
                lstat_count=2,
            )
            result = TsconfigResolver(repo).resolve_alias("@/secret", str(importer))

        assert state["swapped"]
        assert result == str(safe_target)

    def test_extends_swap_after_precheck_reads_verified_inode(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        outside = tmp_path / "outside"
        repo.mkdir()
        outside.mkdir()
        importer = repo / "app.ts"
        importer.write_text("import '@/secret'\n", encoding="utf-8")
        (repo / "tsconfig.json").write_text(
            json.dumps({"extends": "./base.json", "compilerOptions": {}}),
            encoding="utf-8",
        )
        _write_config(repo, "base.json", {"@/*": ["safe/*"]})
        _write_config(outside, "base.json", {"@/*": ["evil/*"]})
        safe_target = repo / "safe" / "secret.ts"
        evil_target = repo / "evil" / "secret.ts"
        safe_target.parent.mkdir()
        evil_target.parent.mkdir()
        safe_target.write_text("export const safe = 1\n", encoding="utf-8")
        evil_target.write_text("export const evil = 1\n", encoding="utf-8")

        with monkeypatch.context() as context:
            state = _swap_after_path_precheck(
                repo / "base.json",
                outside / "base.json",
                context,
                lstat_count=2,
            )
            result = TsconfigResolver(repo).resolve_alias("@/secret", str(importer))

        assert state["swapped"]
        assert result == str(safe_target)

    def test_alias_swap_after_precheck_has_no_path_reprobe(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        outside = tmp_path / "outside"
        repo.mkdir()
        outside.mkdir()
        importer = repo / "app.ts"
        importer.write_text("import '@/secret'\n", encoding="utf-8")
        _write_config(repo, "tsconfig.json", {"@/*": ["src/*"]})
        target = repo / "src" / "secret.ts"
        target.parent.mkdir()
        target.write_text("export const safe = 1\n", encoding="utf-8")
        external_target = outside / "secret.ts"
        external_target.write_text("export const evil = 1\n", encoding="utf-8")
        original_is_file = Path.is_file

        with monkeypatch.context() as context:
            state = _swap_after_path_precheck(target, external_target, context, lstat_count=1)

            def guarded_is_file(path):
                assert not (Path(path) == target and state["swapped"]), (
                    "alias pathname was re-probed after verification"
                )
                return original_is_file(path)

            context.setattr(Path, "is_file", guarded_is_file)
            result = TsconfigResolver(repo).resolve_alias("@/secret", str(importer))

        assert state["swapped"]
        assert result == str(target)

    @pytest.mark.parametrize(
        "attack",
        ["file_path", "base_url", "paths", "extends", "symlink"],
    )
    def test_zero_external_os_access(self, tmp_path, monkeypatch, attack):
        repo = tmp_path / "repo"
        outside = tmp_path / "outside"
        repo.mkdir()
        outside.mkdir()
        importer = (outside if attack == "file_path" else repo) / "app.ts"
        importer.write_text("import '@/secret'\n", encoding="utf-8")
        outside_target = outside / "secret.ts"
        outside_target.write_text("export const secret = 1\n", encoding="utf-8")

        if attack == "file_path":
            _write_config(repo, "tsconfig.json", {"@/*": ["src/*"]})
        elif attack == "base_url":
            _write_config(repo, "tsconfig.json", {"@/*": ["*"]}, "../outside")
        elif attack == "paths":
            _write_config(repo, "tsconfig.json", {"@/*": ["../outside/*"]})
        elif attack == "extends":
            (outside / "base.json").write_text(
                json.dumps({"compilerOptions": {"paths": {"@/*": ["*"]}}}),
                encoding="utf-8",
            )
            (repo / "tsconfig.json").write_text(
                json.dumps({"extends": "../outside/base.json", "compilerOptions": {}}),
                encoding="utf-8",
            )
        else:
            _write_config(repo, "tsconfig.json", {"@/*": ["src/*"]})
            source_dir = repo / "src"
            source_dir.mkdir()
            (source_dir / "secret.ts").symlink_to(outside_target)

        resolver = TsconfigResolver(repo)
        with monkeypatch.context() as context:
            readlink_calls = _guard_external_os_access(repo, context)
            result = resolver.resolve_alias("@/secret", str(importer))

        assert result is None
        assert readlink_calls == []

    def test_missing_boundary_fails_closed(self, tmp_path):
        importer = tmp_path / "src" / "app.ts"
        importer.parent.mkdir()
        importer.write_text("import '@/target'\n", encoding="utf-8")
        _write_config(tmp_path, "tsconfig.json", {"@/*": ["src/*"]})
        target = tmp_path / "src" / "target.ts"
        target.write_text("export const target = 1\n", encoding="utf-8")

        assert TsconfigResolver().resolve_alias("@/target", str(importer)) is None

    def test_contained_alias_resolves(self, tmp_path):
        importer = tmp_path / "src" / "app.ts"
        importer.parent.mkdir()
        importer.write_text("import '@/target'\n", encoding="utf-8")
        _write_config(tmp_path, "tsconfig.json", {"@/*": ["src/*"]})
        target = tmp_path / "src" / "target.ts"
        target.write_text("export const target = 1\n", encoding="utf-8")

        result = TsconfigResolver(tmp_path).resolve_alias("@/target", str(importer))

        assert result == str(target.resolve())

    @pytest.mark.parametrize(
        ("base_url", "replacement"),
        [("../outside", "*"), (".", "../outside/*")],
    )
    def test_escaping_base_url_or_path_is_refused(
        self,
        tmp_path,
        base_url,
        replacement,
    ):
        repo = tmp_path / "repo"
        outside = tmp_path / "outside"
        repo.mkdir()
        outside.mkdir()
        importer = repo / "app.ts"
        importer.write_text("import '@escape/secret'\n", encoding="utf-8")
        (outside / "secret.ts").write_text("export const secret = 1\n", encoding="utf-8")
        _write_config(
            repo,
            "tsconfig.json",
            {"@escape/*": [replacement]},
            base_url=base_url,
        )

        result = TsconfigResolver(repo).resolve_alias("@escape/secret", str(importer))

        assert result is None

    def test_escaping_extends_is_refused(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        outside_config = tmp_path / "outside.json"
        outside_config.write_text(
            json.dumps({"compilerOptions": {"paths": {"@/*": ["repo/src/*"]}}}),
            encoding="utf-8",
        )
        (repo / "tsconfig.json").write_text(
            json.dumps({"extends": "../outside.json", "compilerOptions": {}}),
            encoding="utf-8",
        )
        importer = repo / "app.ts"
        importer.write_text("import '@/target'\n", encoding="utf-8")
        target = repo / "src" / "target.ts"
        target.parent.mkdir()
        target.write_text("export const target = 1\n", encoding="utf-8")

        assert TsconfigResolver(repo).resolve_alias("@/target", str(importer)) is None

    def test_no_outside_filesystem_probe_occurs(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        outside = tmp_path / "outside"
        repo.mkdir()
        outside.mkdir()
        importer = repo / "app.ts"
        importer.write_text("import '@escape/secret'\n", encoding="utf-8")
        _write_config(repo, "tsconfig.json", {"@escape/*": ["../outside/*"]})
        original_is_file = Path.is_file
        original_is_dir = Path.is_dir
        original_read_text = Path.read_text

        def assert_contained(path, operation):
            resolved = path.resolve()
            assert resolved == repo or repo in resolved.parents, (
                f"{operation} probed outside repository: {resolved}"
            )

        def guarded_is_file(path):
            assert_contained(path, "is_file")
            return original_is_file(path)

        def guarded_is_dir(path):
            assert_contained(path, "is_dir")
            return original_is_dir(path)

        def guarded_read_text(path, *args, **kwargs):
            assert_contained(path, "read_text")
            return original_read_text(path, *args, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(Path, "is_file", guarded_is_file)
            context.setattr(Path, "is_dir", guarded_is_dir)
            context.setattr(Path, "read_text", guarded_read_text)
            result = TsconfigResolver(repo).resolve_alias("@escape/secret", str(importer))

        assert result is None

    def test_no_outside_extends_probe_occurs(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        outside_config = tmp_path / "outside.json"
        outside_config.write_text(
            json.dumps({"compilerOptions": {"paths": {"@/*": ["src/*"]}}}),
            encoding="utf-8",
        )
        (repo / "tsconfig.json").write_text(
            json.dumps({"extends": "../outside.json", "compilerOptions": {}}),
            encoding="utf-8",
        )
        importer = repo / "app.ts"
        importer.write_text("import '@/target'\n", encoding="utf-8")
        original_is_file = Path.is_file
        original_read_text = Path.read_text

        def assert_contained(path, operation):
            resolved = path.resolve()
            assert resolved == repo or repo in resolved.parents, (
                f"{operation} probed outside repository: {resolved}"
            )

        def guarded_is_file(path):
            assert_contained(path, "is_file")
            return original_is_file(path)

        def guarded_read_text(path, *args, **kwargs):
            assert_contained(path, "read_text")
            return original_read_text(path, *args, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(Path, "is_file", guarded_is_file)
            context.setattr(Path, "read_text", guarded_read_text)
            result = TsconfigResolver(repo).resolve_alias("@/target", str(importer))

        assert result is None

    def test_symlinked_alias_target_outside_repository_is_refused(self, tmp_path):
        repo = tmp_path / "repo"
        outside = tmp_path / "outside"
        repo.mkdir()
        outside.mkdir()
        importer = repo / "app.ts"
        importer.write_text("import '@/secret'\n", encoding="utf-8")
        _write_config(repo, "tsconfig.json", {"@/*": ["src/*"]})
        outside_target = outside / "secret.ts"
        outside_target.write_text("export const secret = 1\n", encoding="utf-8")
        source_dir = repo / "src"
        source_dir.mkdir()
        (source_dir / "secret.ts").symlink_to(outside_target)

        result = TsconfigResolver(repo).resolve_alias("@/secret", str(importer))

        assert result is None

    def test_code_parser_threads_repository_boundary(self, tmp_path):
        parser = CodeParser(repo_root=tmp_path)

        assert parser._tsconfig_resolver.repo_root == tmp_path.resolve()
