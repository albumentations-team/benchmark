from pathlib import Path

from augbench.dataset_access import prewarm_files, reordered_items


def test_prewarm_reads_every_file_once_in_selection_order(tmp_path: Path) -> None:
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    first.write_bytes(b"abc")
    second.write_bytes(b"de")

    result = prewarm_files((first, second))

    assert result.files == 2
    assert result.bytes_read == 5


def test_reordered_items_uses_the_same_seeded_indices() -> None:
    items = tuple(f"item-{index}" for index in range(10_000))

    ordered = reordered_items(items, required_items=8_448, seed=137)

    assert len(ordered) == 8_448
    assert len(set(ordered)) == 8_448
    assert ordered != items[:8_448]
