from pathlib import Path

import pytest
from matplotlib.animation import FFMpegWriter, PillowWriter

from experiments.animate import FPS, writer_for


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
