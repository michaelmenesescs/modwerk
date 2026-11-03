"""RESONATOR -- a four-voice chord resonator insert for FX2."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "tools"))
import resonator_ref as ref  # noqa: E402

from remix.schema import (Gate, Category, Proof, BusRole, Claims, DspSection, Formatter,
                          Harness, Kind, MenuEntry, Module, Param, YBase)

P, S, W = Formatter.PLAIN, Formatter.STEPPED, Formatter.WIDE_STEPPED
CHORD_LABELS = tuple(name for name, _ in ref.CHORDS)
VOIC_LABELS = tuple(name for name, _ in ref.VOICINGS)
SEMI = tuple(f"{(i - 24):+d}" for i in range(49))
NOTE_LABELS = tuple(f"{['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][n % 12]}{n // 12 - 1}"
                    for n in range(36, 36 + 60))

MODULE = Module(
    name="resonator",
    key="RESONATOR",
    kind=Kind.DSP_EFFECT,
    category=Category.TRACK,
    author="michaelmenesescs", author_url="https://github.com/michaelmenesescs",
    proof=Proof.RENDER, proof_note="tools/verify_dsp.py in dsp56kEmu; not on hardware",
    doc="Four tuned feedback combs on a chord: dub-techno stabs and metallic rings.",
    menu=MenuEntry(fx2_id=0x07, donor_desc=0x400d58b8, abbr=b"RESO", fullname=b"Resonator"),
    params=(
        Param(b"TUNE", 14, count=ref.TUNE_COUNT, active=True, formatter=W, labels=NOTE_LABELS,
              doc="Root note C2..B6; with CHRD it sets the four combs' pitches"),
        Param(b"DCAY", 90, active=True, formatter=P,
              doc="Ring time T60, 50 ms at 0 to 20 s at 127; every note rings for the same time"),
        Param(b"BRIT", 80, active=True, formatter=P,
              doc="Damping: 127 keeps the highs, low values make high notes die sooner"),
        Param(b"SPRD", 64, active=True, formatter=P,
              doc="Stereo spread of the four voices, centre to wide"),
        Param(b"DRIV", 0, active=True, formatter=P,
              doc="Drive into the rings; high values hit the ring clip for industrial tones"),
        Param(b"MIX", 0, active=True, formatter=P,
              doc="Dry/wet; 0 is an exact passthrough, 127 fully wet"),
        Param(b"CHRD", 11, count=len(CHORD_LABELS), active=True, formatter=W, labels=CHORD_LABELS,
              doc="Chord the four voices ring at; USER takes INT2..INT4"),
        Param(b"VOIC", 0, count=len(VOIC_LABELS), active=True, formatter=W, labels=VOIC_LABELS,
              doc="Voicing: close, three inversions, drop-2, open or wide"),
        Param(b"INT2", 27, count=49, active=True, formatter=W, labels=SEMI,
              doc="USER chord: second voice, -24..+24 semitones"),
        Param(b"INT3", 31, count=49, active=True, formatter=W, labels=SEMI,
              doc="USER chord: third voice, -24..+24 semitones"),
        Param(b"INT4", 34, count=49, active=True, formatter=W, labels=SEMI,
              doc="USER chord: fourth voice, -24..+24 semitones"),
        Param(b"FREZ", 0, count=2, active=True, formatter=S, labels=("OFF", "ON"),
              doc="Freeze: closes the input and holds the rings losslessly"),
    ),
    dsp=DspSection(asm="modules/resonator/resonator.asm", priority=20,
                   bus_role=BusRole.NONE, ybase=YBase.NEVER, ptable=tuple(ref.ptable())),
    claims=Claims(stock_instance_buffer=True, buffer_words=8192),
    harness=Harness(layout_char="Z", is_server=False),
    gates=(Gate("modules/resonator/verify.py", remix_arg=False),),
    dear={"DCAY": 127, "DRIV": 127, "BRIT": 127, "SPRD": 127, "MIX": 127, "FREZ": 0},
)
