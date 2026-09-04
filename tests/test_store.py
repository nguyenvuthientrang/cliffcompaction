"""PrefixStore bounds: size is the binding limit, entry count a backstop."""

from cliffcompaction.config import Config
from cliffcompaction.engine import Engine
from cliffcompaction.store import Entry, PrefixStore, entry_size


def an_entry(chars: int, cut: int = 10) -> Entry:
    return Entry(head_len=2, summary={"role": "user", "content": "x" * chars}, cut=cut)


def test_put_tracks_total_size():
    store = PrefixStore()
    e = an_entry(1000)
    store.put("h1", e)
    assert store.nbytes == entry_size(e.summary)
    assert store.nbytes >= 1000
    assert e.size == store.nbytes


def test_replacing_a_key_does_not_double_count():
    store = PrefixStore()
    store.put("h1", an_entry(5000))
    store.put("h1", an_entry(1000))
    assert len(store) == 1
    assert store.nbytes < 2000


def test_evicts_least_recently_used_until_under_byte_budget():
    store = PrefixStore(max_bytes=10_000)
    for i in range(6):
        store.put(f"h{i}", an_entry(3000))
    assert store.nbytes <= 10_000
    assert len(store) == 3
    # Oldest gone, newest kept.
    assert store.get("h0") is None
    assert store.get("h5") is not None


def test_get_refreshes_recency_so_a_hot_entry_survives():
    store = PrefixStore(max_bytes=10_000)
    for i in range(3):
        store.put(f"h{i}", an_entry(3000))
    store.get("h0")  # h1 is now the least recently used
    store.put("h9", an_entry(3000))
    assert store.get("h0") is not None
    assert store.get("h1") is None


def test_entry_count_still_bounds_when_entries_are_tiny():
    store = PrefixStore(max_entries=4, max_bytes=10**9)
    for i in range(20):
        store.put(f"h{i}", an_entry(10))
    assert len(store) == 4
    assert store.get("h19") is not None


def test_an_oversized_entry_is_kept_rather_than_evicting_to_empty():
    store = PrefixStore(max_bytes=100)
    store.put("big", an_entry(50_000))
    assert len(store) == 1
    assert store.get("big") is not None
    assert store.nbytes > 100


def test_a_second_oversized_put_drops_the_first():
    store = PrefixStore(max_bytes=100)
    store.put("big1", an_entry(50_000))
    store.put("big2", an_entry(50_000))
    assert len(store) == 1
    assert store.get("big1") is None
    assert store.get("big2") is not None


def test_entry_size_never_raises_on_unserializable_summaries():
    assert entry_size({"content": object()}) == 0


def test_engine_keeps_an_empty_store_it_was_handed():
    """An empty PrefixStore is falsy (it defines __len__), so `store or ...`
    would silently swap it for a default-configured one."""
    mine = PrefixStore(max_entries=7, max_bytes=1234)
    engine = Engine(Config(), store=mine)
    assert engine.store is mine
    assert engine.store._max_bytes == 1234


def test_engine_builds_its_store_from_config():
    engine = Engine(Config(store_max_entries=9, store_max_bytes=4321))
    assert engine.store._max == 9
    assert engine.store._max_bytes == 4321


def test_default_budget_holds_a_realistic_working_set():
    """One entry runs ~0.25 chars per est token of the threshold, so a 210k
    threshold gives ~50k chars. A 32-parallel run needs far less than this."""
    store = PrefixStore()
    for i in range(200):
        store.put(f"h{i}", an_entry(50_000))
    assert len(store) == 200
