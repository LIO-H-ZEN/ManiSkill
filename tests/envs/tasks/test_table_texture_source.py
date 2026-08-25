from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from mani_skill.envs.tasks.pick_anything.randomization.table_randomizer import (
    TableTextureSource,
)


def test_safe_texture_conversion_is_atomic_under_concurrency(tmp_path):
    source_path = tmp_path / "texture.jpg"
    Image.new("RGB", (2048, 1024), color=(20, 40, 60)).save(source_path)
    source = TableTextureSource(max_texture_dim=128)

    with ThreadPoolExecutor(max_workers=32) as executor:
        outputs = list(executor.map(source._ensure_safe_size, [source_path] * 64))

    assert len(set(outputs)) == 1
    safe_path = tmp_path / "_safe" / "texture.png"
    assert outputs[0] == str(safe_path)
    with Image.open(safe_path) as image:
        image.verify()
    with Image.open(safe_path) as image:
        assert image.size == (128, 64)
    assert list(safe_path.parent.glob(".*.tmp")) == []


def test_safe_texture_conversion_repairs_corrupt_cached_file(tmp_path):
    source_path = tmp_path / "texture.jpg"
    Image.new("RGB", (64, 64), color=(20, 40, 60)).save(source_path)
    safe_path = tmp_path / "_safe" / "texture.png"
    safe_path.parent.mkdir()
    safe_path.write_bytes(b"partial png")

    output = TableTextureSource(max_texture_dim=32)._ensure_safe_size(source_path)

    assert output == str(safe_path)
    with Image.open(safe_path) as image:
        image.verify()
    with Image.open(safe_path) as image:
        assert image.size == (32, 32)
