import pytest

from multihorizon_vector_spherical_operator import parse_horizons


def test_parse_horizons_accepts_fixed_positive_unique_leads():
    assert parse_horizons("3,6,12") == (3, 6, 12)


@pytest.mark.parametrize("text", ["", "0,3", "3,-1", "3,3"])
def test_parse_horizons_rejects_invalid_leads(text):
    with pytest.raises(ValueError):
        parse_horizons(text)
