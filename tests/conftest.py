import shutil
import sys

import lamindb as ln
import pytest


def pytest_sessionstart():
    ln.setup.init(
        storage="./test-scdataloaderdb", name="test-scdataloader", modules="bionty"
    )
    ln.settings.creation.artifact_silence_missing_run_warning = True
    ln.settings.track_run_inputs = False


def pytest_sessionfinish(session):
    shutil.rmtree("./test-scdataloaderdb")
    ln.setup.delete("test-scdataloader", force=True)


## each test runs on cwd to its temp dir
# @pytest.fixture(autouse=True)
# def go_to_tmpdir(request):
#    # Get the fixture dynamically by its name.
#    tmpdir = request.getfixturevalue("tmpdir")
#    # ensure local test created packages can be imported
#    sys.path.insert(0, str(tmpdir))
#    # Chdir only for the duration of the test.
#    with tmpdir.as_cwd():
#        yield
