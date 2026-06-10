import json

from scripts import fetch_houseexpo


def test_resolve_target_names_are_train_plus_holdout():
    cfg = {"maps": {"train": ["a", "b"], "holdout": ["c"]}}
    assert fetch_houseexpo.curated_names(cfg) == ["a", "b", "c"]


def test_stamp_sha_replaces_sentinel(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text('maps:\n  dataset_sha: "PINNED_AT_FETCH"\n', encoding="utf-8")
    fetch_houseexpo.stamp_sha(str(cfg_file), "a" * 40)
    text = cfg_file.read_text(encoding="utf-8")
    assert "PINNED_AT_FETCH" not in text
    assert "a" * 40 in text


def test_copy_curated_is_idempotent(tmp_path):
    src = tmp_path / "full" / "json"
    src.mkdir(parents=True)
    (src / "room_single.json").write_text(json.dumps({"verts": [[0, 0]]}), encoding="utf-8")
    dst = tmp_path / "maps"
    n1 = fetch_houseexpo.copy_curated(str(src), str(dst), ["room_single"])
    n2 = fetch_houseexpo.copy_curated(str(src), str(dst), ["room_single"])
    assert n1 == 1 and n2 == 1
    assert (dst / "room_single.json").exists()
