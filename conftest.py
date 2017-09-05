import pytest
from tempfile import TemporaryDirectory
from os import chdir

@pytest.fixture
def cleandir():
    """cd to a clean tempdir for tests"""
    tempdir = TemporaryDirectory()
    chdir(tempdir.name)
    return tempdir
