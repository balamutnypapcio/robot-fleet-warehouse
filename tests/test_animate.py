import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

from experiments.animate import LAYOUTS, build_animation, save_animation, simulate  # noqa: E402
from fleet.layout import load_layout  # noqa: E402


def make_animation():
    layout = load_layout(LAYOUTS / "mini.yaml")
    model, orders, result = simulate(layout, n_orders=2, seed=1, horizon=10.0)
    # High speed keeps the number of frames (and the test time) small.
    return build_animation(layout, model, orders, result, speed=200.0)


def test_export_gif(tmp_path):
    _, anim = make_animation()
    path = tmp_path / "run.gif"
    save_animation(anim, path)
    assert path.read_bytes()[:6] in (b"GIF87a", b"GIF89a")


@pytest.mark.skipif(not FFMpegWriter.isAvailable(), reason="ffmpeg not installed")
def test_export_mp4(tmp_path):
    _, anim = make_animation()
    path = tmp_path / "run.mp4"
    save_animation(anim, path)
    assert b"ftyp" in path.read_bytes()[:16]
