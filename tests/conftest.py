import pytest
from databend_aiserver.runtime import detect_runtime

@pytest.fixture(autouse=True, scope="session")
def init_runtime():
    detect_runtime()
