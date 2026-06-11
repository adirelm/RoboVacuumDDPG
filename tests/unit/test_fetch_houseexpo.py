import io
import json
import tarfile

from scripts import _fetch_archive, fetch_houseexpo


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


def test_copy_curated_by_id_maps_real_id_to_logical_name(tmp_path):
    # Real dataset files are named <id>.json; the curated copy renames them to
    # the logical config names (room_single, ...).
    src = tmp_path / "json"
    src.mkdir(parents=True)
    (src / "deadbeef.json").write_text(json.dumps({"verts": [[0, 0]]}), encoding="utf-8")
    dst = tmp_path / "maps"
    n = fetch_houseexpo.copy_curated_by_id(str(src), str(dst), {"room_single": "deadbeef"})
    assert n == 1
    assert (dst / "room_single.json").exists()
    assert not (dst / "deadbeef.json").exists()


def test_archive_url_builds_pinned_raw_github_url():
    url = _fetch_archive.archive_url(
        "https://github.com/TeaganLi/HouseExpo", "a" * 40, "HouseExpo/json.tar.gz"
    )
    assert url == (
        "https://raw.githubusercontent.com/TeaganLi/HouseExpo/" + "a" * 40 + "/HouseExpo/json.tar.gz"
    )


def test_extract_json_flattens_and_counts_real_members(tmp_path):
    # Build a tiny gzip tar mirroring the real json/<id>.json layout in memory.
    archive = tmp_path / "json.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        payload = json.dumps({"verts": [[0, 0]], "id": "abc"}).encode()
        info = tarfile.TarInfo("json/abc.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
        # a non-json member must be ignored
        readme = b"not json"
        rinfo = tarfile.TarInfo("json/readme.txt")
        rinfo.size = len(readme)
        tar.addfile(rinfo, io.BytesIO(readme))
    out = tmp_path / "json"
    n = _fetch_archive.extract_json(str(archive), str(out))
    assert n == 1
    assert (out / "abc.json").exists()  # flattened (json/ prefix stripped)
    assert not (out / "readme.txt").exists()
