import unittest

import torch

from spherical_operator_experiment import (
    RealSphericalHarmonicTransform,
    SphericalMarkovOperator,
    SphericalPoleOperator,
)


class RealSphericalHarmonicTransformTests(unittest.TestCase):
    def test_bandlimited_round_trip(self):
        transform = RealSphericalHarmonicTransform(nlat=12, nlon=24, lmax=5)
        coefficients = torch.randn(3, transform.n_modes)
        recovered = transform.analysis(transform.synthesis(coefficients))
        torch.testing.assert_close(recovered, coefficients, atol=2e-5, rtol=2e-5)

    def test_resolution_transfer_preserves_coefficients(self):
        low = RealSphericalHarmonicTransform(nlat=12, nlon=24, lmax=5)
        high = RealSphericalHarmonicTransform(nlat=24, nlon=48, lmax=5)
        coefficients = torch.randn(2, low.n_modes)
        high_field = high.synthesis(low.analysis(low.synthesis(coefficients)))
        recovered = high.analysis(high_field)
        torch.testing.assert_close(recovered, coefficients, atol=3e-5, rtol=3e-5)


class SphericalOperatorTests(unittest.TestCase):
    def setUp(self):
        self.transform = RealSphericalHarmonicTransform(nlat=8, nlon=16, lmax=3)
        self.previous = torch.randn(2, self.transform.n_modes)
        self.drivers = torch.randn(2, 5)

    def test_both_operators_conserve_global_mean(self):
        for model in (
            SphericalMarkovOperator(5, self.transform.mode_degrees),
            SphericalPoleOperator(5, self.transform.mode_degrees),
        ):
            if isinstance(model, SphericalPoleOperator):
                state = model.initialize(self.previous)
                predicted, _ = model.step(self.previous, self.drivers, state)
            else:
                predicted, _ = model.step(self.previous, self.drivers)
            torch.testing.assert_close(predicted[:, 0], self.previous[:, 0])

    def test_pole_full_sequence_matches_repeated_step(self):
        model = SphericalPoleOperator(5, self.transform.mode_degrees)
        drivers = torch.randn(2, 7, 5)
        full = model.predict_sequence(self.previous, drivers)
        previous = self.previous
        state = model.initialize(previous)
        outputs = []
        for t in range(drivers.shape[1]):
            previous, state = model.step(previous, drivers[:, t], state)
            outputs.append(previous)
        repeated = torch.stack(outputs, dim=1)
        torch.testing.assert_close(full, repeated)

    def test_fixed_poles_are_not_trainable_or_weight_decayed(self):
        model = SphericalPoleOperator(5, self.transform.mode_degrees)
        names = dict(model.named_parameters())
        self.assertNotIn("pole_decay", names)
        self.assertIn("pole_gate_logits", names)


if __name__ == "__main__":
    unittest.main()
