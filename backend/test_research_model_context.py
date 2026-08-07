import unittest

from research_model_context import (
    cascadia_nsaf_advisories,
    is_great_cascadia_candidate,
    research_model_context,
)
from grade3_earth import dipole_field, igrf14_degree1_coefficients


class CascadiaCompoundHazardTests(unittest.TestCase):
    def test_great_shallow_offshore_cascadia_event_is_screened(self):
        event = {
            "id": "test-cascadia",
            "mag": 9.0,
            "lat": 44.0,
            "lon": -125.5,
            "depth": 20.0,
        }
        self.assertTrue(is_great_cascadia_candidate(event))
        advisory = cascadia_nsaf_advisories([event])[0]
        self.assertEqual(advisory["target"], "northern San Andreas fault")
        self.assertFalse(advisory["active"])
        self.assertEqual(advisory["status"], "PENDING_AUTHORITATIVE_CONFIRMATION")
        self.assertIsNone(advisory["probability"])
        self.assertIsNone(advisory["timing_window"])

    def test_moderate_event_does_not_create_compound_advisory(self):
        event = {"mag": 6.5, "lat": 44.0, "lon": -125.5, "depth": 20.0}
        self.assertFalse(is_great_cascadia_candidate(event))
        self.assertEqual(cascadia_nsaf_advisories([event]), [])

    def test_distant_great_event_does_not_create_cascadia_advisory(self):
        event = {"mag": 8.5, "lat": 37.0, "lon": 142.0, "depth": 25.0}
        self.assertFalse(is_great_cascadia_candidate(event))

    def test_research_updates_do_not_modify_global_seismic_ratios(self):
        context = research_model_context()
        self.assertFalse(context["global_seismic_zone_ratios_modified"])
        self.assertFalse(context["mendocino_triple_junction_geometry"]["probability_change"])
        feature = context["mendocino_triple_junction_geometry"]["feature"]
        self.assertEqual(feature["geometry"]["type"], "MultiPoint")
        self.assertEqual(len(feature["geometry"]["coordinates"]), 27)
        self.assertEqual(len(feature["properties"]["depths_km"]), 27)
        self.assertFalse(context["solar_helicity"]["probability_change"])
        self.assertFalse(context["inner_core"]["warning_change"])


class Igrf14FallbackTests(unittest.TestCase):
    def test_igrf14_epoch_and_secular_variation(self):
        self.assertEqual(igrf14_degree1_coefficients(2025.0), (-29350.0, -1410.3, 4545.5))
        self.assertEqual(igrf14_degree1_coefficients(2030.0), (-29287.0, -1360.3, 4438.0))

    def test_tilted_dipole_has_physical_north_and_east_components(self):
        north, east, down = dipole_field(0.0, 0.0, 2025.0)
        self.assertGreater(north, 0.0)
        self.assertNotEqual(east, 0.0)
        self.assertNotEqual(down, 0.0)


if __name__ == "__main__":
    unittest.main()
