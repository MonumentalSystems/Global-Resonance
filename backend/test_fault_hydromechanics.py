import unittest

try:
    from .fault_hydromechanics import (
        effective_normal_stress_change_mpa,
        gofar_reference_payload,
        jellyball_hydromechanics_payload,
        pore_pressure_response,
        strengthening_from_pressure_drop,
    )
except ImportError:
    from fault_hydromechanics import (
        effective_normal_stress_change_mpa,
        gofar_reference_payload,
        jellyball_hydromechanics_payload,
        pore_pressure_response,
        strengthening_from_pressure_drop,
    )


class FaultHydromechanicsTests(unittest.TestCase):
    def test_positive_pore_pressure_reduces_effective_stress(self):
        self.assertEqual(effective_normal_stress_change_mpa(2.5), -2.5)
        response = pore_pressure_response(2.5)
        self.assertEqual(response["nucleation_tendency"], "promoting")

    def test_pressure_drop_strengthens_fault_without_inventing_rate(self):
        result = strengthening_from_pressure_drop(15, 50)
        self.assertEqual(result["effective_normal_stress_increase_mpa"], 15)
        self.assertEqual(result["strengthening_pct"], 30)
        self.assertIsNone(result["rupture_rate_multiplier"])

    def test_gofar_scenario_is_labeled_as_local_and_modeled(self):
        gofar = gofar_reference_payload()
        self.assertIn("not a global calibration", gofar["scope"])
        self.assertIn("conceptual simulation", gofar["dynamic_cycle"]["calibration"])
        self.assertEqual(gofar["structural_barrier"]["stepover_offset_m"], [100, 400])

    def test_global_zone_ratios_are_not_recalibrated(self):
        payload = jellyball_hydromechanics_payload()
        self.assertFalse(payload["global_zone_ratios_modified"])

    def test_invalid_strengthening_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            strengthening_from_pressure_drop(-1, 50)
        with self.assertRaises(ValueError):
            strengthening_from_pressure_drop(1, 0)


if __name__ == "__main__":
    unittest.main()
