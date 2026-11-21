"""Versioned outputs use the frozen producers and never replace existing results."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib as mpl
import pytest

from experiments.paper.figures import render_versioned as versioned
from figure_rendering_helpers import render_profile


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = json.loads((ROOT / "tests/fixtures/offline_figure_rendering_before_2b.json").read_text())
PRODUCERS = {
    "primary-response": "plot_paper120_primary_boxplot_v1",
    "primary-parameters": "plot_paper120_primary_parameter_boxplot_v1",
    "sbi-response": "plot_paper120_sbi_response_boxplot_v1",
    "terminal-domain": "plot_paper120_terminal_domain_ablation_v1",
}


@pytest.mark.parametrize(("figure", "producer"), PRODUCERS.items())
def test_versioned_images_match_frozen_producer_outputs(figure, producer, offline_figure_root, monkeypatch):
    root = offline_figure_root
    expected = EXPECTED["renderers"][producer]["fallback_files"]
    update = mpl.RcParams.update

    def fallback(parameters, *args, **kwargs):
        update(parameters, *args, **kwargs)
        parameters["text.usetex"] = False
        parameters["font.serif"] = ["DejaVu Serif"]

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    with mpl.rc_context():
        mpl.rcdefaults()
        monkeypatch.setattr(mpl.RcParams, "update", fallback)
        destination = versioned.render_versioned(figure, "v1", root)
    assert destination == root.resolve() / "results/paper/figures" / figure / "v1"
    assert {p.name for p in destination.iterdir()} == set(expected)
    assert all(p.stat().st_size > 0 for p in destination.iterdir())
    if render_profile() != EXPECTED["profile"]:
        pytest.skip("stored fallback image bytes require the recorded Matplotlib/font profile")
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir()} == expected


@pytest.mark.parametrize("version", ["../manuscript", "/tmp", "v0", "v1/child", "v1.pdf"])
def test_version_token_cannot_address_another_location(tmp_path, version):
    with pytest.raises(ValueError, match="version such as v1"):
        versioned.render_versioned("primary-response", version, tmp_path)
    assert not (tmp_path / "results").exists()


def test_existing_result_and_link_to_manuscript_are_refused(tmp_path):
    root = tmp_path
    figure = "primary-response"
    directory = root / "results/paper/figures" / figure
    (directory / "v1").mkdir(parents=True)
    old = directory / "v1/plot.pdf"
    old.write_bytes(b"accepted")
    with pytest.raises(FileExistsError):
        versioned.render_versioned(figure, "v1", root)
    assert old.read_bytes() == b"accepted"
    manuscript = root / "docs/paper/latex/figures"
    manuscript.mkdir(parents=True)
    try:
        (directory / "v2").symlink_to(manuscript, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symbolic link support unavailable: {error}")
    with pytest.raises(ValueError, match="symbolic link"):
        versioned.render_versioned(figure, "v2", root)
    assert list(manuscript.iterdir()) == []


def test_input_failure_does_not_publish_an_empty_version(tmp_path):
    with pytest.raises(FileNotFoundError):
        versioned.render_versioned("primary-response", "v2", tmp_path)
    assert not (tmp_path / "results").exists()


def test_concurrent_publisher_cannot_overwrite_existing_result(tmp_path, monkeypatch):
    def competing_render(figure, root, staging):
        paths = (staging / "plot.pdf", staging / "plot.png")
        for path in paths:
            path.write_bytes(b"new")
        winner = root / "results/paper/figures" / figure / "v2"
        winner.mkdir(parents=True)
        (winner / "plot.pdf").write_bytes(b"winner")
        return paths

    monkeypatch.setattr(versioned, "_render", competing_render)
    with pytest.raises(FileExistsError):
        versioned.render_versioned("sbi-response", "v2", tmp_path)
    winner = tmp_path / "results/paper/figures/sbi-response/v2"
    assert [(p.name, p.read_bytes()) for p in winner.iterdir()] == [("plot.pdf", b"winner")]


def test_help_requires_no_scientific_or_plotting_packages(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c",
         "import runpy,sys; sys.path.insert(0,sys.argv[1]);sys.argv=['render_versioned','--help'];"
         "runpy.run_module('experiments.paper.figures.render_versioned',run_name='__main__')", str(ROOT)],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert all(figure in completed.stdout for figure in PRODUCERS)
    assert not list(tmp_path.iterdir())


def test_incomplete_file_copy_is_removed_and_can_be_retried(tmp_path, monkeypatch):
    def checked_images(figure, root, staging):
        files = (staging / "plot.pdf", staging / "plot.png")
        for file in files:
            file.write_bytes(b"checked")
        return files

    copy = versioned.shutil.copyfileobj

    def fail_second_copy(source, destination):
        if Path(source.name).suffix == ".png":
            destination.write(b"partial")
            raise OSError("simulated write interruption")
        copy(source, destination)

    monkeypatch.setattr(versioned, "_render", checked_images)
    with monkeypatch.context() as patch:
        patch.setattr(versioned.shutil, "copyfileobj", fail_second_copy)
        with pytest.raises(OSError, match="write interruption"):
            versioned.render_versioned("sbi-response", "v3", tmp_path)
    result = versioned.render_versioned("sbi-response", "v3", tmp_path)
    assert {p.name: p.read_bytes() for p in result.iterdir()} == {
        "plot.pdf": b"checked",
        "plot.png": b"checked",
    }
