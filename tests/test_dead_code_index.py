"""The in-memory call-target index must answer exactly what the old LIKE queries answered."""

from pathlib import Path

import pytest

from code_review_graph import refactor
from code_review_graph.graph import GraphStore
from code_review_graph.parser import EdgeInfo, NodeInfo
from code_review_graph.refactor import _CallTargetIndex, find_dead_code

_CALL_TARGETS = [
    "pkg/mod.py::snake_case_fn",
    "pkg/mod.py::Snake_Case_Fn",
    "pkg/mod.py::snakeXcaseXfn",
    "Service::handle",
    "Service.handle",
    "other::Service.handle",
    "bare_name",
    "pkg/a.py::helper",
    "pkg/a.py::helper2",
    "pkg/%odd::helper",
    "Widget.render",
    "pkg/x.py::Widget_Factory.build",
]


@pytest.fixture
def store(tmp_path: Path):
    graph = GraphStore(str(tmp_path / "g.db"))
    for index, target in enumerate(_CALL_TARGETS):
        graph.upsert_edge(
            EdgeInfo(
                kind="CALLS",
                source=f"/repo/c{index}.py::caller",
                target=target,
                file_path=f"/repo/c{index}.py",
                line=1,
            )
        )
    graph.upsert_edge(
        EdgeInfo(
            kind="REFERENCES",
            source="/repo/r.py::ref",
            target="pkg/r.py::helper",
            file_path="/repo/r.py",
            line=1,
        )
    )
    graph.commit()
    yield graph
    graph.close()


def _sql_suffix_ids(store: GraphStore, name: str) -> set[int]:
    rows = store._conn.execute(
        "SELECT id FROM edges WHERE kind = 'CALLS' AND target_qualified LIKE ?",
        (f"%::{name}",),
    ).fetchall()
    return {row["id"] for row in rows}


def _sql_member_hit(store: GraphStore, *fragments: str) -> bool:
    count = store._conn.execute(
        "SELECT COUNT(*) FROM edges WHERE kind = 'CALLS'"
        " AND (target_qualified LIKE ? OR target_qualified LIKE ?)",
        (f"%{fragments[0]}%", f"%{fragments[1]}%"),
    ).fetchone()[0]
    return count > 0


@pytest.mark.parametrize(
    "name",
    [
        "helper", "HELPER", "snake_case_fn", "Snake_Case_Fn",
        "handle", "Handle", "nothing", "h_lper", "%lper",
    ],
)
def test_suffix_rows_equal_like_query(store: GraphStore, name: str) -> None:
    index = _CallTargetIndex(store._conn)
    assert {row["id"] for row in index.suffix_rows(name)} == _sql_suffix_ids(store, name)


@pytest.mark.parametrize(
    "fragments",
    [
        ("Service.", "Service."),
        ("/repo/x.py::Widget.", "Widget."),
        ("Widget_Factory.", "Nope."),
        ("Widget.", "Nope."),
        ("Missing.", "Nope."),
        ("Widget_Factor_.", "Nope."),
        ("Wi_get.", "Nope."),
    ],
)
def test_member_fragments_equal_like_query(store: GraphStore, fragments: tuple[str, str]) -> None:
    index = _CallTargetIndex(store._conn)
    assert index.has_target_containing(*fragments) == _sql_member_hit(store, *fragments)


class _SqlReferenceIndex:
    """The pre-index implementation: one LIKE query per lookup."""

    def __init__(self, conn) -> None:
        self._conn = conn

    def suffix_rows(self, name: str):
        return self._conn.execute(
            "SELECT * FROM edges WHERE kind = 'CALLS' AND target_qualified LIKE ?",
            (f"%::{name}",),
        ).fetchall()

    def has_target_containing(self, member_prefix: str, bare_prefix: str) -> bool:
        return (
            self._conn.execute(
                "SELECT COUNT(*) FROM edges WHERE kind = 'CALLS'"
                " AND (target_qualified LIKE ? OR target_qualified LIKE ?)",
                (f"%{member_prefix}%", f"%{bare_prefix}%"),
            ).fetchone()[0]
            > 0
        )


def _seed_nodes(store: GraphStore) -> None:
    for path, language in (("/repo/a.py", "python"), ("/repo/b.py", "python")):
        store.upsert_node(
            NodeInfo(
                kind="File", name=path, file_path=path,
                line_start=1, line_end=99, language=language,
            )
        )
    for kind, name, path in (
        ("Function", "helper", "/repo/a.py"),
        ("Function", "snake_case_fn", "/repo/a.py"),
        ("Function", "never_called", "/repo/b.py"),
        ("Class", "Widget", "/repo/b.py"),
        ("Class", "Service", "/repo/a.py"),
        ("Class", "Unused", "/repo/b.py"),
    ):
        store.upsert_node(
            NodeInfo(
                kind=kind, name=name, file_path=path,
                line_start=2, line_end=9, language="python",
            )
        )
    store.commit()


def test_find_dead_code_identical_to_sql_implementation(
    store: GraphStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_nodes(store)
    fast = find_dead_code(store)
    monkeypatch.setattr(refactor, "_CallTargetIndex", _SqlReferenceIndex)
    reference = find_dead_code(store)
    assert fast, "fixture must produce dead symbols, otherwise equality proves nothing"
    assert any(item["name"] == "never_called" for item in fast)
    assert fast == reference
