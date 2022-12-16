import pytest


@pytest.fixture(params=[1, 2, 4])
def num_ifos(request):
    return request.param
