import pytest

from src.data.split import load, make, restore, save


@pytest.mark.parametrize("row", ["a.png\n", "a.png,a.png,extra\n", ",a.png\n", "a.png,\n"])
def test_load_rejects_malformed_csv_rows(tmp_path, row):
    (tmp_path / "train.csv").write_text("image,mask\n" + row, encoding="utf-8")
    (tmp_path / "valid.csv").write_text("image,mask\nb.png,b.png\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Invalid split row"):
        load(tmp_path)


def test_save_preserves_csv_format_filenames_and_pair_order(tmp_path):
    splits = {
        "train": [("old/image/c.jpg", "old/mask/c.png"),
                  ("old/image/b.png", "old/mask/b.png")],
        "valid": [("old/image/a.png", "old/mask/a.png")],
    }

    save(splits, tmp_path)

    assert (tmp_path / "train.csv").read_bytes() == b"image,mask\r\nc.jpg,c.png\r\nb.png,b.png\r\n"
    assert (tmp_path / "valid.csv").read_bytes() == b"image,mask\r\na.png,a.png\r\n"
    assert load(tmp_path) == {
        "train": [("c.jpg", "c.png"), ("b.png", "b.png")],
        "valid": [("a.png", "a.png")],
    }


def test_load_existing_csv_with_quoted_filenames_and_blank_lines(tmp_path):
    (tmp_path / "train.csv").write_bytes(b'image,mask\r\n\r\n"a,1.jpg","a,1.png"\r\n')
    (tmp_path / "valid.csv").write_bytes(b"image,mask\r\nb.png,b.png\r\n")

    assert load(tmp_path) == {
        "train": [("a,1.jpg", "a,1.png")],
        "valid": [("b.png", "b.png")],
    }


@pytest.mark.parametrize("serialized", [False, True])
def test_restore_relocated_pairs_preserves_membership_and_order(tmp_path, serialized):
    splits = {
        "train": [("old/image/c.jpg", "old/mask/c.png"),
                  ("old/image/b.png", "old/mask/b.png")],
        "valid": [("old/image/a.png", "old/mask/a.png")],
    }
    if serialized:
        save(splits, tmp_path)
        splits = load(tmp_path)
    pairs = [(f"new/image/{name}.{ext}", f"new/mask/{name}.png")
             for name, ext in (("a", "png"), ("b", "png"), ("c", "jpg"))]

    assert restore(pairs, splits) == {
        "train": [pairs[2], pairs[1]],
        "valid": [pairs[0]],
    }


@pytest.mark.parametrize("splits, error", [
    ({"train": [("a.jpg", "a.png")]}, "incomplete"),
    ({"train": [], "valid": [("b.jpg", "b.png")]}, "empty"),
    ({"train": [("a.jpg", "a.png")], "valid": [("a.jpg", "a.png")]}, "duplicate"),
    ({"train": [("a.jpg", "a.png")], "valid": [("missing.jpg", "missing.png")]}, "missing"),
    ({"train": [("a.jpg", "a.png")], "valid": [("b.jpg", "b.png")]}, "does not match"),
])
def test_restore_rejects_split_corruption_and_dataset_drift(splits, error):
    pairs = [(f"image/{name}.jpg", f"mask/{name}.png") for name in ("a", "b", "c")]

    with pytest.raises(RuntimeError, match=error):
        restore(pairs, splits)


@pytest.mark.parametrize("ratio, valid_count", [(0.01, 1), (0.5, 2), (0.99, 3)])
def test_make_keeps_pairs_together_without_mutating_input(ratio, valid_count):
    pairs = [(f"image/{name}.jpg", f"mask/{name}.png") for name in ("a", "b", "c", "d")]
    original = list(pairs)

    splits = make(pairs, ratio)

    assert pairs == original
    assert len(splits["valid"]) == valid_count
    assert len(splits["train"]) == len(pairs) - valid_count
    assert set(splits["train"]).isdisjoint(splits["valid"])
    assert sorted(splits["train"] + splits["valid"]) == original
