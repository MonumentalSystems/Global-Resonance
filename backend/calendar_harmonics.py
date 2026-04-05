#!/usr/bin/env python3
"""
Ancient Calendars as Subharmonic Detectors

The lunar-solar calendar reconciliation problem IS the detection
of incommensurate frequencies in a coupled oscillator system.

KEY: The Egyptian Sothic cycle (1461yr) = 0.994x Bond (0.6% error).
The Egyptians were tracking the Bond cycle through Sirius drift.
Bond was rediscovered by Gerard Bond in 1997 from ice-rafting.
"""


def main():
    bond = 1470
    nodal = 18.61

    cycles = [
        ("Metonic (Greek/Babylon/China)", 19.0),
        ("Saros (Babylonian)", 18.03),
        ("Callippic (Greek)", 76.0),
        ("Maya Calendar Round", 52.0),
        ("Sothic (Egyptian)", 1461.0),
        ("Hindu Yuga subcycle", 432.0),
        ("Vedic Maha Yuga", 4320.0),
        ("Great Year (Platonic)", 25772.0),
        ("Nodal precession", 18.61),
    ]

    print("Calendar cycles vs subharmonic ladder:")
    print(f"{'Cycle':35s} {'Period':>8s} {'Bond':>8s} {'Nodal':>8s}")
    for name, T in cycles:
        bond_r = T / bond
        nodal_r = T / nodal
        marks = []
        if abs(bond_r - round(bond_r)) < 0.05 and round(bond_r) > 0:
            marks.append(f"{round(bond_r)}xBond")
        if abs(nodal_r - round(nodal_r)) < 0.3 and round(nodal_r) > 0:
            marks.append(f"{round(nodal_r)}xNodal")
        m = " ".join(marks)
        print(f"  {name:33s} {T:7.0f}yr {bond_r:7.3f}x {nodal_r:7.1f}x  {m}")

    print(f"\nSothic/Bond = {1461/bond:.4f} (0.6% error)")
    print(f"Nodal x 79 = {nodal*79:.1f}yr (Bond = {bond}yr, 0.01% error)")


if __name__ == "__main__":
    main()
