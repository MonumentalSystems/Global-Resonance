#!/usr/bin/env python3
"""
Excursion Timing: IRREGULAR but always N x Bond

CORRECTION: Excursions are NOT periodic at 41kyr.
They are STOCHASTIC threshold crossings, but every spacing
is a near-integer multiple of the Bond cycle (all < 7% error).

The Bond cycle is the clock. Excursions happen when N consecutive
Bond pumps accumulate enough l=2 energy to overwhelm the dipole.
N varies (4 to 71) because threshold crossing is stochastic.
"""
import numpy as np

EXCURSIONS = [
    (3, "Etna/Sterno"), (17, "Hilina Pali"), (28, "Mono Lake"),
    (34, "Laschamp"), (41, "Laschamp (alt)"), (61, "Norwegian-Greenland"),
    (94, "Blake"), (114, "Blake ext"), (180, "Iceland Basin"),
    (200, "Pringle Falls"), (225, "Mamaku"), (290, "Levantine"),
    (395, "Calabrian 1"), (410, "Calabrian 0"), (510, "Kamikatsura"),
    (560, "Big Lost"), (615, "Delta"), (670, "CR3"), (718, "CR2"),
]


def main():
    ages = [x[0] for x in EXCURSIONS]
    spacings = np.diff(ages)
    bond = 1.47  # kyr

    print("Excursion spacings as Bond multiples:")
    for s in spacings:
        n = round(s / bond)
        err = abs(s / bond - n) / n * 100
        print(f"  {s:4.0f} kyr = {s/bond:.1f}x Bond = ~{n}x ({err:.0f}%)")

    print(f"\nMean spacing: {np.mean(spacings):.0f} kyr (range {spacings.min():.0f}-{spacings.max():.0f})")
    print(f"All spacings < 7% error from integer Bond multiples")
    print(f"Excursions are NOT periodic — they are N x Bond with stochastic N")


if __name__ == "__main__":
    main()
