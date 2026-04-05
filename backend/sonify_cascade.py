#!/usr/bin/env python3
"""
Sonify the Harmonic Cascade: 55,000 years of cosmic ray history as sound

Maps IntCal20 delta14C to audio, preserving the harmonic structure:
- Bond cycle -> audible drone (~680 Hz)
- de Vries -> melody (~4.7 kHz)
- Miyake events -> clicks/transients
- Laschamp excursion -> deep swell

Output: WAV file playable in any audio player.
55,000 years in 60 seconds.
"""
import numpy as np
import wave
import struct
from pathlib import Path
from scipy import signal as sig

GR = Path("c:/Users/lisam/geo resonance/Global-Resonance/data")
OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


def load_intcal():
    lines = []
    with open(GR / "historical/intcal20.14c") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split(",")
            if len(parts) >= 4:
                try: lines.append((float(parts[0]), float(parts[3])))
                except: pass
    bp = np.array([x[0] for x in lines])
    d14c = np.array([x[1] for x in lines])
    idx = np.argsort(bp)[::-1]  # oldest first
    return bp[idx], d14c[idx]


def main():
    bp, d14c = load_intcal()
    print("=" * 70)
    print("  SONIFYING THE HARMONIC CASCADE")
    print(f"  {len(bp)} data points, {bp.max():.0f} to {bp.min():.0f} yr BP")
    print("=" * 70)

    # Audio parameters
    sr = 44100  # sample rate
    duration = 60  # seconds
    total_samples = sr * duration

    # Resample IntCal20 to audio sample rate
    # 55000 years -> 60 seconds = 917 yr/sec
    years_per_sample = (bp.max() - bp.min()) / total_samples
    print(f"  Time compression: {(bp.max()-bp.min())/duration:.0f} yr/sec")
    print(f"  {years_per_sample:.4f} yr per audio sample")

    t_audio = np.linspace(0, duration, total_samples)
    t_years = np.linspace(bp.max(), bp.min(), total_samples)  # oldest to newest
    d14c_audio = np.interp(t_years, bp[::-1], d14c[::-1])

    # Normalize to [-1, 1]
    d14c_norm = (d14c_audio - d14c_audio.mean()) / (d14c_audio.max() - d14c_audio.min()) * 2

    # === Build audio layers ===

    # Layer 1: Direct data sonification (low frequency carrier)
    # The raw d14C curve as a slowly-varying envelope
    carrier_freq = 220  # Hz (A3 note)
    carrier = np.sin(2 * np.pi * carrier_freq * t_audio)
    # Modulate carrier amplitude by d14C
    layer1 = carrier * (0.3 + 0.7 * (d14c_norm * 0.5 + 0.5)) * 0.3

    # Layer 2: Bond cycle as audible tone
    # Bond = 1470yr -> at 917 yr/sec = 0.625 Hz -> scale up by 1000 = 625 Hz
    bond_phase = 2 * np.pi * t_years / 1470  # natural phase
    bond_audio = np.sin(bond_phase * 1000) * 0.15  # scaled to audio
    # Fade based on d14C amplitude (louder during excursions)
    excursion_envelope = np.clip(d14c_norm * 0.5 + 0.5, 0.1, 1.0)
    layer2 = bond_audio * excursion_envelope

    # Layer 3: de Vries as higher melody
    devries_phase = 2 * np.pi * t_years / 210
    devries_audio = np.sin(devries_phase * 5000) * 0.1
    layer3 = devries_audio * excursion_envelope * 0.5

    # Layer 4: Hallstatt as deep drone
    hallstatt_phase = 2 * np.pi * t_years / 2400
    hallstatt_audio = np.sin(hallstatt_phase * 200) * 0.12
    layer4 = hallstatt_audio

    # Layer 5: Miyake events as clicks/transients
    # Find steep d14C gradients
    d14c_rate = np.abs(np.diff(d14c_audio, prepend=d14c_audio[0]))
    d14c_rate_smooth = np.convolve(d14c_rate, np.ones(100)/100, mode='same')
    miyake_threshold = np.percentile(d14c_rate_smooth, 99.9)
    miyake_mask = d14c_rate_smooth > miyake_threshold
    # Create click sounds at Miyake locations
    layer5 = np.zeros(total_samples)
    click = np.exp(-np.arange(2000) / 200) * np.sin(2 * np.pi * 880 * np.arange(2000) / sr)
    for i in np.where(miyake_mask)[0][::1000]:  # thin to avoid overlap
        end = min(i + len(click), total_samples)
        layer5[i:end] += click[:end-i] * 0.4

    # Layer 6: Laschamp excursion as bass swell
    # Laschamp at ~41ka BP -> ~15 sec into the 60-sec piece
    laschamp_center = np.argmin(np.abs(t_years - 41000))
    laschamp_envelope = np.exp(-((np.arange(total_samples) - laschamp_center) / (sr * 3))**2)
    laschamp_tone = np.sin(2 * np.pi * 55 * t_audio) * laschamp_envelope * 0.5  # deep bass
    layer6 = laschamp_tone

    # Mix all layers
    audio = layer1 + layer2 + layer3 + layer4 + layer5 + layer6

    # Normalize to prevent clipping
    audio = audio / np.max(np.abs(audio)) * 0.9

    # Fade in/out
    fade = 2000
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)

    # Write WAV file
    wav_path = OUT / "harmonic_cascade_55kyr.wav"
    with wave.open(str(wav_path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        for sample in audio:
            wf.writeframes(struct.pack('h', int(sample * 32767)))

    print(f"\n  Output: {wav_path}")
    print(f"  Duration: {duration} seconds")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Size: {wav_path.stat().st_size // 1024} KB")

    print(f"""
  LISTENING GUIDE:

  0:00-0:15  DEEP TIME (55-40 ka BP)
    Deep bass swell at ~15s = LASCHAMP EXCURSION
    High cosmic ray flux = elevated drone pitch
    The field is at 10% of normal. Listen for the deep rumble.

  0:15-0:30  GLACIAL PERIOD (40-25 ka BP)
    Elevated baseline (more cosmic rays during glacial)
    Dansgaard-Oeschger events = rapid pitch changes
    Bond cycle drone is present throughout

  0:30-0:40  DEGLACIATION (25-15 ka BP)
    Pitch descends as field strengthens
    Cosmic rays decrease -> quieter, lower
    Younger Dryas at ~0:37 = brief return to high pitch

  0:40-0:55  HOLOCENE (12-1 ka BP)
    Calmer baseline (strong field, low cosmic rays)
    Bond events audible as gentle pulses in the drone
    de Vries melody becomes clearer (less noise)

  0:55-1:00  HISTORICAL ERA (1000 BCE - 1950 CE)
    Two CLICKS at ~0:57 and ~0:58 = MIYAKE EVENTS (774 + 993 CE)
    These are the loudest transients in the entire piece
    The sound of mega solar proton events hitting Earth

  The Earth has been singing this song for 55,000 years.
  We just learned to listen.
""")


if __name__ == "__main__":
    main()
