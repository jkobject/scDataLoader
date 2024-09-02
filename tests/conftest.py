import sys
import pytest
import lamindb_setup as ln_setup


@pytest.fixture(scope="session")
def setup_instance():
    ln_setup.init(storage="./testdb")
    yield
    ln_setup.delete("testdb", force=True)


# each test runs on cwd to its temp dir
@pytest.fixture(autouse=True)
def go_to_tmpdir(request):
    # Get the fixture dynamically by its name.
    tmpdir = request.getfixturevalue("tmpdir")
    # ensure local test created packages can be imported
    sys.path.insert(0, str(tmpdir))
    # Chdir only for the duration of the test.
    with tmpdir.as_cwd():
        yield
