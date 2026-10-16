"""TypeScript tsconfig.json / jsconfig.json path alias resolver.

Resolves TypeScript path aliases (e.g., ``@/ -> src/``) declared in
``compilerOptions.paths`` so that ``IMPORTS_FROM`` edges can point to
real file paths instead of raw alias strings. Plain-JS projects (Vue,
Nuxt, Vite) declare the same aliases in ``jsconfig.json``, which shares
the ``compilerOptions`` schema, so it is handled by the same parser.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Extensions probed when resolving an alias target
_PROBE_EXTENSIONS = [".ts", ".tsx", ".js", ".jsx", ".vue", ".mts", ".cts"]

# Config filenames to look for when walking up the directory tree.
# Order defines precedence within a directory: tsconfig.json wins over
# tsconfig.app.json, and both win over jsconfig.json (issue #776).
# jsconfig.json is a tsconfig.json with JS-oriented compiler defaults;
# none of those implicit defaults affect baseUrl/paths resolution, so
# the same parser (JSONC + relative "extends" chains) covers it.
# The nearest directory containing any of these names still wins over
# configs higher up the tree, matching editor/bundler behavior.
_TSCONFIG_NAMES = ["tsconfig.json", "tsconfig.app.json", "jsconfig.json"]


def _path_is_within(path: Path, root: Path) -> bool:
    """Return whether a lexically normalized path is inside the repository root."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _lexical_absolute(path: Path) -> Path:
    """Normalize a path without querying filesystem components."""
    return Path(os.path.abspath(os.path.normpath(os.fspath(path))))


def _safe_lexical_path(path: Path, root: Path) -> Optional[Path]:
    """Return a lexically contained path without querying the filesystem."""
    normalized = _lexical_absolute(path)
    return normalized if _path_is_within(normalized, root) else None


@contextmanager
def _open_repo_path(
    path: Path,
    root: Path,
    *,
    directory: bool = False,
) -> Iterator[Optional[int]]:
    """Open a contained path component-by-component without following symlinks."""
    normalized = _safe_lexical_path(path, root)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if normalized is None or nofollow is None or directory_flag is None:
        yield None
        return

    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | nofollow | directory_flag | close_on_exec
    file_flags = os.O_RDONLY | nofollow | close_on_exec
    opened: list[int] = []
    try:
        current_fd = os.open(root, directory_flags)
        opened.append(current_fd)
        parts = normalized.relative_to(root).parts
        for part in parts[:-1]:
            current_fd = os.open(part, directory_flags, dir_fd=current_fd)
            opened.append(current_fd)
        if parts:
            target_flags = directory_flags if directory else file_flags
            target_fd = os.open(parts[-1], target_flags, dir_fd=current_fd)
            opened.append(target_fd)
        else:
            target_fd = current_fd
    except OSError:
        target_fd = None
    try:
        yield target_fd
    finally:
        for fd in reversed(opened):
            try:
                os.close(fd)
            except OSError as exc:
                logger.debug("TsconfigResolver: cannot close descriptor %s: %s", fd, exc)


def _read_repo_text(path: Path, root: Path) -> Optional[str]:
    """Read a regular file through a descriptor rooted at the repository."""
    with _open_repo_path(path, root) as fd:
        if fd is None or not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        chunks: list[bytes] = []
        while chunk := os.read(fd, 64 * 1024):
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


def _probe_regular_file(path: Path, root: Path) -> Optional[Path]:
    """Return a lexical file identity after descriptor-relative verification."""
    normalized = _safe_lexical_path(path, root)
    if normalized is None:
        return None
    with _open_repo_path(normalized, root) as fd:
        if fd is None or not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
    return normalized


def _probe_directory(path: Path, root: Path) -> Optional[Path]:
    """Return a lexical directory identity after descriptor-relative verification."""
    normalized = _safe_lexical_path(path, root)
    if normalized is None:
        return None
    with _open_repo_path(normalized, root, directory=True) as fd:
        if fd is None or not stat.S_ISDIR(os.fstat(fd).st_mode):
            return None
    return normalized


class TsconfigResolver:
    """Resolve TypeScript path aliases within an explicit repository boundary."""

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self._repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self._cache: dict[str, Optional[dict]] = {}

    @property
    def repo_root(self) -> Optional[Path]:
        """Return the immutable repository containment boundary."""
        return self._repo_root

    def _contained_path(self, path: Path) -> Optional[Path]:
        """Return a safe lexical path within the repository boundary."""
        if self._repo_root is None:
            logger.warning(
                "TsconfigResolver: refusing %s because no repo_root boundary was set",
                path,
            )
            return None
        safe_path = _safe_lexical_path(path, self._repo_root)
        if safe_path is None:
            logger.warning(
                "TsconfigResolver: refusing path outside repository root: %s",
                _lexical_absolute(path),
            )
        return safe_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_alias(self, import_str: str, file_path: str) -> Optional[str]:
        """Resolve a TS path alias to an absolute file path, or None."""
        try:
            config = self._load_tsconfig_for_file(file_path)
            if config is None:
                return None

            base_url: Optional[str] = config.get("baseUrl")
            paths: dict[str, list[str]] = config.get("paths", {})
            tsconfig_dir: str = config.get("_tsconfig_dir", "")

            if not paths:
                return None

            base_dir = self._contained_path(Path(tsconfig_dir) / (base_url or ""))
            if base_dir is None:
                return None
            return self._match_and_probe(import_str, paths, base_dir)
        except (OSError, ValueError, TypeError):
            logger.debug(
                "TsconfigResolver: unexpected error for %s", file_path, exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_tsconfig_for_file(self, file_path: str) -> Optional[dict]:
        """Find and load tsconfig.json for the given file."""
        start_dir = self._contained_path(Path(file_path).parent)
        if start_dir is None:
            return None
        current = start_dir
        visited: list[str] = []

        while True:
            dir_str = str(current)
            if dir_str in self._cache:
                result = self._cache[dir_str]
                for visited_dir in visited:
                    self._cache[visited_dir] = result
                return result

            visited.append(dir_str)

            for name in _TSCONFIG_NAMES:
                candidate = self._contained_path(current / name)
                if candidate is not None:
                    config = self._parse_tsconfig(candidate)
                else:
                    config = None
                if config is not None:
                    config["_tsconfig_dir"] = dir_str
                    for visited_dir in visited:
                        self._cache[visited_dir] = config
                    return config

            parent = current.parent
            if parent == current or current == self._repo_root:
                for visited_dir in visited:
                    self._cache[visited_dir] = None
                return None
            current = parent

    def _parse_tsconfig(self, tsconfig_path: Path) -> Optional[dict]:
        """Parse a tsconfig.json file (supports JSONC comments)."""
        if self._repo_root is None:
            return None
        contained_tsconfig = self._contained_path(tsconfig_path)
        if contained_tsconfig is None:
            return None
        raw = _read_repo_text(contained_tsconfig, self._repo_root)
        if raw is None:
            return None
        seen: set[str] = set()
        return self._resolve_extends(contained_tsconfig, seen, raw)

    def _resolve_extends(
        self,
        tsconfig_path: Path,
        seen: set[str],
        raw: Optional[str] = None,
    ) -> dict:
        """Recursively resolve the tsconfig extends chain."""
        if self._repo_root is None:
            return {}
        contained_tsconfig = self._contained_path(tsconfig_path)
        if contained_tsconfig is None:
            return {}
        tsconfig_path = contained_tsconfig
        canonical = str(tsconfig_path)
        if canonical in seen:
            logger.debug("TsconfigResolver: cycle detected at %s", canonical)
            return {}
        seen = seen | {canonical}

        if raw is None:
            raw = _read_repo_text(tsconfig_path, self._repo_root)
        if raw is None:
            logger.debug("TsconfigResolver: cannot read %s", tsconfig_path)
            return {}

        stripped = self._strip_jsonc_comments(raw)
        try:
            data: dict = json.loads(stripped)
        except json.JSONDecodeError:
            logger.debug("TsconfigResolver: invalid JSON in %s", tsconfig_path)
            return {}

        result: dict = {}

        extends: Optional[str] = data.get("extends")
        if extends and isinstance(extends, str) and extends.startswith("."):
            parent_path = tsconfig_path.parent / extends
            if not parent_path.suffix:
                parent_path = parent_path.with_suffix(".json")
            contained_parent = self._contained_path(parent_path)
            if contained_parent is not None:
                parent_config = self._resolve_extends(contained_parent, seen)
                parent_opts = parent_config.get("compilerOptions", {})
                result.setdefault("compilerOptions", {}).update(parent_opts)

        child_opts: dict = data.get("compilerOptions", {})
        result.setdefault("compilerOptions", {}).update(child_opts)

        compiler_options = result.get("compilerOptions", {})
        if "baseUrl" in compiler_options:
            result["baseUrl"] = compiler_options["baseUrl"]
        if "paths" in compiler_options:
            result["paths"] = compiler_options["paths"]

        return result

    def _strip_jsonc_comments(self, text: str) -> str:
        """Remove // and /* */ comments and trailing commas from JSONC."""
        result: list[str] = []
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]

            if ch == '"':
                result.append(ch)
                i += 1
                while i < n:
                    c = text[i]
                    result.append(c)
                    if c == "\\" and i + 1 < n:
                        i += 1
                        result.append(text[i])
                    elif c == '"':
                        break
                    i += 1
                i += 1
                continue

            if ch == "/" and i + 1 < n and text[i + 1] == "*":
                i += 2
                while i < n - 1:
                    if text[i] == "*" and text[i + 1] == "/":
                        i += 2
                        break
                    i += 1
                else:
                    i = n
                continue

            if ch == "/" and i + 1 < n and text[i + 1] == "/":
                i += 2
                while i < n and text[i] != "\n":
                    i += 1
                continue

            result.append(ch)
            i += 1

        stripped = "".join(result)
        stripped = re.sub(r",\s*([\]}])", r"\1", stripped)
        return stripped

    def _match_and_probe(
        self,
        import_str: str,
        paths: dict[str, list[str]],
        base_dir: Path,
    ) -> Optional[str]:
        """Match import_str against alias patterns and probe the filesystem."""
        def _pattern_specificity(item: tuple[str, list[str]]) -> int:
            pat = item[0]
            return len(pat.partition("*")[0])

        sorted_paths = sorted(paths.items(), key=_pattern_specificity, reverse=True)

        for pattern, replacements in sorted_paths:
            suffix = _match_pattern(pattern, import_str)
            if suffix is None:
                continue

            for replacement in replacements:
                if "*" in replacement:
                    mapped = replacement.replace("*", suffix, 1)
                else:
                    mapped = replacement

                candidate_base = self._contained_path(base_dir / mapped)
                if candidate_base is None:
                    continue
                found = _probe_path(candidate_base, self._repo_root)
                if found:
                    return str(found)

        return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _match_pattern(pattern: str, import_str: str) -> Optional[str]:
    """Return the wildcard-matched suffix if pattern matches import_str."""
    if "*" not in pattern:
        return "" if import_str == pattern else None

    prefix, _, suffix_pat = pattern.partition("*")
    if not (import_str.startswith(prefix) and import_str.endswith(suffix_pat)):
        return None

    end = len(import_str) - len(suffix_pat) if suffix_pat else len(import_str)
    return import_str[len(prefix):end]


def _probe_path(base: Path, repo_root: Optional[Path]) -> Optional[Path]:
    """Probe a path only after each candidate is contained in the repository."""
    if repo_root is None:
        return None

    def contained_file(candidate: Path) -> Optional[Path]:
        return _probe_regular_file(candidate, repo_root)

    if found := contained_file(base):
        return found
    for ext in _PROBE_EXTENSIONS:
        candidate = base.with_suffix(ext) if not base.suffix else Path(str(base) + ext)
        if found := contained_file(candidate):
            return found
    safe_base = _probe_directory(base, repo_root)
    if safe_base is not None:
        for ext in _PROBE_EXTENSIONS:
            if found := contained_file(safe_base / f"index{ext}"):
                return found
    return None
