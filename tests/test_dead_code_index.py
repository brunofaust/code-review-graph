"""The on-disk call-target indexes must answer what the old LIKE queries answered."""

from pathlib import Path

import pytest

from code_review_graph import migrations
from code_review_graph.graph import GraphStore
from code_review_graph.incremental import full_build, incremental_update
from code_review_graph.migrations import CALL_TAIL_EXPR
from code_review_graph.parser import EdgeInfo, NodeInfo
from code_review_graph.refactor import find_dead_code

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
    "name", ["helper", "HELPER", "Handle", "handle", "nothing", "bare_name", "render", "build"]
)
def test_tail_lookup_equals_like_query(store: GraphStore, name: str) -> None:
    got = {e.id for e in store.get_call_edges_by_tail(name)}
    assert got == _sql_suffix_ids(store, name)


def test_tail_lookup_treats_underscore_and_percent_literally(store: GraphStore) -> None:
    # The old LIKE read ``_`` as a wildcard and also matched snakeXcaseXfn.
    assert len(_sql_suffix_ids(store, "snake_case_fn")) == 3
    assert len(store.get_call_edges_by_tail("snake_case_fn")) == 2
    assert store.get_call_edges_by_tail("%lper") == []
    assert store.get_call_edges_by_tail("h_lper") == []


def test_bare_target_is_not_a_suffix_match(store: GraphStore) -> None:
    assert store.get_call_edges_by_tail("bare_name") == []
    assert store.get_call_edges_by_tail("Widget.render") == []


def test_tail_lookup_uses_the_expression_index(store: GraphStore) -> None:
    plan = " ".join(
        str(row[-1])
        for row in store._conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM edges WHERE kind = 'CALLS' "
            f"AND {CALL_TAIL_EXPR} = lower(?)",
            ("helper",),
        )
    )
    assert "idx_edges_call_tail" in plan, plan


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
        ("w", "zzzz"),
        ("t.", "zzzz"),
        ("Q", "zz"),
        ("zz", "yy"),
    ],
)
def test_member_fragments_equal_like_query(store: GraphStore, fragments: tuple[str, str]) -> None:
    store.refresh_call_target_index()
    assert store.has_call_target_like(*fragments) == _sql_member_hit(store, *fragments)


def _trigram_targets(store: GraphStore) -> set[str]:
    return {r[0] for r in store._conn.execute("SELECT target FROM call_target_trigram")}


def test_trigram_refresh_tracks_edges(tmp_path: Path) -> None:
    graph = GraphStore(str(tmp_path / "old.db"))
    graph.upsert_edge(
        EdgeInfo(kind="CALLS", source="/r/a.py::f", target="m::gone", file_path="/r/a.py", line=1)
    )
    graph.commit()
    assert _trigram_targets(graph) == set()  # a bare upsert does not refresh
    graph.refresh_call_target_index()
    assert _trigram_targets(graph) == {"m::gone"}
    graph._conn.execute("DELETE FROM edges")
    graph.commit()
    graph.refresh_call_target_index()
    assert _trigram_targets(graph) == set()
    assert not graph.has_call_target_like("gone")
    graph.close()


def test_migration_v14_fills_the_trigram_table(tmp_path: Path) -> None:
    graph = GraphStore(str(tmp_path / "v13.db"))
    graph.upsert_edge(
        EdgeInfo(kind="CALLS", source="/r/a.py::f", target="m::kept", file_path="/r/a.py", line=1)
    )
    graph.commit()
    assert _trigram_targets(graph) == set()
    migrations._migrate_v14(graph._conn)
    graph.commit()
    assert _trigram_targets(graph) == {"m::kept"}
    assert [e.target_qualified for e in graph.get_call_edges_by_tail("kept")] == ["m::kept"]
    assert migrations.LATEST_VERSION == 14
    graph.close()


def test_trigram_follows_full_build_and_incremental_update(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def alpha():\n    return 1\n")
    (repo / "b.py").write_text("from a import alpha\n\ndef beta():\n    return alpha()\n")
    graph = GraphStore(str(tmp_path / "g.db"))
    full_build(repo, graph)
    assert graph.has_call_target_like("alpha")
    assert not graph.has_call_target_like("gamma_call")

    (repo / "b.py").write_text("def beta():\n    return gamma_call()\n")
    incremental_update(repo, graph, changed_files=["b.py"])
    assert graph.has_call_target_like("gamma_call")
    assert not graph.has_call_target_like("alpha")  # removed target disappears

    (repo / "b.py").write_text("def beta():\n    return 2\n")
    incremental_update(repo, graph, changed_files=["b.py"])
    assert not graph.has_call_target_like("gamma_call")
    graph.close()


class _SqlReferenceStore:
    """The pre-index implementation: one LIKE query per lookup."""

    @staticmethod
    def get_call_edges_by_tail(self: GraphStore, name: str):
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE kind = 'CALLS' AND target_qualified LIKE ?",
            (f"%::{name}",),
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    @staticmethod
    def has_call_target_like(self: GraphStore, *fragments: str) -> bool:
        return any(
            self._conn.execute(
                "SELECT COUNT(*) FROM edges WHERE kind = 'CALLS' AND target_qualified LIKE ?",
                (f"%{fragment}%",),
            ).fetchone()[0]
            for fragment in fragments
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
    store.refresh_call_target_index()
    fast = find_dead_code(store)
    monkeypatch.setattr(
        GraphStore, "get_call_edges_by_tail", _SqlReferenceStore.get_call_edges_by_tail
    )
    monkeypatch.setattr(
        GraphStore, "has_call_target_like", _SqlReferenceStore.has_call_target_like
    )
    reference = find_dead_code(store)
    assert fast, "fixture must produce dead symbols, otherwise equality proves nothing"
    assert any(item["name"] == "never_called" for item in fast)
    assert fast == reference
