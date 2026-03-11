import pytest

from tt.base.instantiate.graph import Graph


@pytest.fixture()
def graph() -> "Graph[str]":
    return Graph(lambda s: s)


def test_lookup_missing_node_returns_none(graph: "Graph[str]"):
    assert graph.lookup("ddd") is None


def test_inserts_nodes_when_not_there_yet(graph: "Graph[str]"):
    assert graph.lookup("ddd") is None

    node = graph.lookup_or_insert_node("ddd")
    assert node.data == "ddd"

    looked_up = graph.lookup("ddd")
    assert looked_up is not None
    assert looked_up.data == "ddd"


def test_lookup_or_insert_returns_same_node_instance(graph: "Graph[str]"):
    first = graph.lookup_or_insert_node("x")
    second = graph.lookup_or_insert_node("x")
    assert first is second


def test_can_remove_nodes_and_get_length(graph: "Graph[str]"):
    assert graph.is_empty()
    assert graph.lookup("ddd") is None

    graph.lookup_or_insert_node("ddd")
    assert not graph.is_empty()

    graph.remove_node("ddd")
    assert graph.lookup("ddd") is None
    assert graph.is_empty()


def test_remove_missing_node_raises_keyerror(graph: "Graph[str]"):
    with pytest.raises(KeyError):
        graph.remove_node("not-there")


def test_roots_simple(graph: "Graph[str]"):
    graph.insert_edges("1", "2")
    roots = graph.roots()
    assert len(roots) == 1
    assert roots[0].data == "2"

    graph.insert_edges("2", "1")
    roots = graph.roots()
    assert len(roots) == 0


def test_roots_complex(graph: "Graph[str]"):
    graph.insert_edges("1", "2")
    graph.insert_edges("1", "3")
    graph.insert_edges("3", "4")

    roots = graph.roots()
    assert len(roots) == 2
    root_data = {n.data for n in roots}
    assert {"2", "4"}.issubset(root_data)


def test_find_cycle_none_on_acyclic(graph: "Graph[str]"):
    graph.insert_edges("A", "B")
    graph.insert_edges("B", "C")
    # No cycle
    assert graph.find_cycle_slow() is None


def test_find_cycle_detects_cycle(graph: "Graph[str]"):
    graph.insert_edges("1", "2")
    graph.insert_edges("2", "3")
    graph.insert_edges("3", "1")

    res = graph.find_cycle_slow()
    assert isinstance(res, str)
    tokens = res.split(" -> ")
    assert len(tokens) >= 2
    # Last token must repeat an earlier token (cycle closed)
    assert tokens[-1] in tokens[:-1]
    # All tokens except the last should be unique
    assert len(tokens[:-1]) == len(set(tokens[:-1]))
    # Optional sanity: all tokens are node keys we inserted
    assert set(tokens).issubset({"1", "2", "3"})


def test_repr_and_to_string_non_empty(graph: "Graph[str]"):
    graph.insert_edges("u", "v")
    s = graph.to_string()
    r = repr(graph)
    assert isinstance(s, str) and s
    assert isinstance(r, str) and r
