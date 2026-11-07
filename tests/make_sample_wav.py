"""Regenerate tests/fixtures/sample.wav: a 0.5 s, 22.05 kHz stereo 440 Hz tone."""

import math
import struct
import wave
from pathlib import Path

RATE = 22050
SECONDS = 0.5
OUT = Path(__file__).parent / "fixtures" / "sample.wav"


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    frames = bytearray()
    for i in range(int(RATE * SECONDS)):
        sample = int(8000 * math.sin(2 * math.pi * 440 * i / RATE))
        frames += struct.pack("<hh", sample, sample)
    with wave.open(str(OUT), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(bytes(frames))


if __name__ == "__main__":
    main()
