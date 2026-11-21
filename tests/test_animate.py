from pathlib import Path

import pytest
from matplotlib.animation import FFMpegWriter, PillowWriter

from experiments.animate import FPS, LAYOUTS, Timeline, simulate, writer_for
from fleet.layout import load_layout
from sim.simulator import read_csv, write_csv


@pytest.mark.parametrize(
    ("name", "writer"),
    [("run.gif", PillowWriter), ("run.mp4", FFMpegWriter), ("RUN.MP4", FFMpegWriter)],
)
def test_writer_for_known_extensions(name, writer):
    w = writer_for(Path(name))
    assert isinstance(w, writer)
    assert w.fps == FPS


def test_writer_for_unknown_extension():
    with pytest.raises(ValueError):
        writer_for(Path("run.avi"))


def test_timeline_from_a_csv_log_ends_in_the_final_state(tmp_path):
    layout = load_layout(LAYOUTS / "mini.yaml")
    model, _, result = simulate(layout, n_orders=4, seed=1, horizon=100)
    path = tmp_path / "run.csv"
    write_csv(result.log, path)
    timeline = Timeline(layout, model, read_csv(path))
    assert timeline.states[-1] == result.state
    assert timeline.end == pytest.approx(result.time, abs=0.001)
