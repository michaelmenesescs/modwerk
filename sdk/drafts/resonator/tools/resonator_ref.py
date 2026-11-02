#!/usr/bin/env python3
"""RESONATOR reference model: the arithmetic the DSP56300 engine must reproduce.

A chord resonator insert for the Octatrack. The track's audio (mono sum, a fixed
low cut) excites four tuned feedback combs whose pitches form a chord:

    in -> HP 60 Hz -> [ comb(root+n1) comb(root+n2) comb(root+n3) comb(root+n4) ] -> pan -> DC block -> MIX

Each comb is a delay line read through a first-order allpass for the fractional
part (lossless, so tuning and decay are exact: a linear read loses highs on
every pass), a one-pole damping lowpass and a cubic soft clip inside the loop.
The loop gain of every voice is derived from one decay TIME and divided by the
damping filter's gain at that voice's pitch, so every note rings for the same
time. FREZ holds the ring: loop gain 1, damping off, input closed, the soft clip
bounds the level.

Controls (12 slots, two pages):
  page 1  TUNE DCAY BRIT SPRD DRIV MIX
  page 2  CHRD VOIC INT2 INT3 INT4 FREZ

This is an original design built from published techniques (feedback comb /
Karplus-Strong resonators, one-pole damping, Jot-style equal-decay loop gain).
No code, tables or sounds from any instrument's firmware.

Pure standard library so it runs anywhere. `python3 resonator_ref.py test`
runs the checks; `python3 resonator_ref.py render OUTDIR` writes WAV demos.
"""
import math
import random
import struct
import sys
import wave

FS = 44100.0

# ---------------------------------------------------------------- controls

TUNE_LOW = 36            # MIDI note of TUNE 0 (C2, 65.4 Hz)
TUNE_COUNT = 60          # TUNE 0..59 -> C2..B6
LOWEST_NOTE = 24         # voices are folded up by octaves until >= C1 (32.7 Hz)
HIGHEST_NOTE = 108       # and down until <= C8

# CHRD: name, four semitone offsets from the root. Order is API once released
# (projects store the index), so new chords go at the end.
CHORDS = [
    ("USER", None),               # INT2..INT4 set notes 2-4
    ("OCT",  (0, 12, 19, 24)),
    ("5TH",  (0, 7, 12, 19)),
    ("MIN",  (0, 3, 7, 12)),
    ("MAJ",  (0, 4, 7, 12)),
    ("MIN7", (0, 3, 7, 10)),
    ("MAJ7", (0, 4, 7, 11)),
    ("DOM7", (0, 4, 7, 10)),
    ("MIN9", (0, 3, 10, 14)),
    ("SUS2", (0, 2, 7, 12)),
    ("SUS4", (0, 5, 7, 12)),
    ("DUB",  (0, 7, 10, 15)),     # minor 7 stacked 1-5-b7-b3: the dub stab
    ("MI11", (0, 7, 10, 17)),
    ("QRTL", (0, 5, 10, 15)),     # quartal
    ("CLST", (0, 1, 6, 7)),       # cluster / tritone: metallic, industrial
]

# VOIC: how the four chord notes are arranged (SH-4D style voicing).
VOICINGS = ["CLSE", "INV1", "INV2", "INV3", "DRP2", "OPEN", "WIDE"]

INT_COUNT = 49           # INT2..INT4: -24..+24 semitones, 24 = 0


def chord_notes(root, chord, voicing, ints=(24 + 3, 24 + 7, 24 + 12)):
    """The four MIDI notes the combs ring at."""
    name, shape = CHORDS[chord]
    if shape is None:
        shape = (0,) + tuple(i - 24 for i in ints)
    notes = sorted(root + s for s in shape)
    v = VOICINGS[voicing]
    if v.startswith("INV"):
        for _ in range(int(v[3])):                 # lowest note up an octave, n times
            notes = sorted(notes[1:] + [notes[0] + 12])
    elif v == "DRP2":                               # second-highest down an octave
        notes = sorted(notes[:2] + [notes[2] - 12] + notes[3:])
    elif v == "OPEN":                               # 2nd and 4th up an octave
        notes = sorted([notes[0], notes[1] + 12, notes[2], notes[3] + 12])
    elif v == "WIDE":                               # spread over three octaves
        notes = sorted([notes[0] - 12, notes[1], notes[2] + 12, notes[3] + 12])
    out = []
    for n in notes:
        while n < LOWEST_NOTE:
            n += 12
        while n > HIGHEST_NOTE:
            n -= 12
        out.append(n)
    return out


def midi_hz(n):
    return 440.0 * 2.0 ** ((n - 69) / 12.0)


def decay_seconds(dcay):
    """DCAY 0..127 -> T60 from 50 ms to 20 s, exponential."""
    return 0.05 * (400.0 ** (dcay / 127.0))


def damping_coef(brit):
    """BRIT 0..127 -> one-pole lowpass pole a (y = (1-a) x + a y1). 127 = open."""
    fc = 400.0 * (2.0 ** (brit / 127.0 * 5.6))       # 400 Hz .. ~19 kHz
    fc = min(fc, 19000.0)
    return math.exp(-2.0 * math.pi * fc / FS) if brit < 127 else 0.0


# ---------------------------------------------------------------- engine


class Resonator:
    """One instance: the state the DSP keeps per track and slot."""

    def __init__(self, **knobs):
        self.k = dict(TUNE=24, DCAY=90, BRIT=80, SPRD=64, DRIV=0, MIX=64,
                      CHRD=11, VOIC=0, INT2=27, INT3=31, INT4=34, FREZ=0)
        self.k.update(knobs)
        self.lines = [[0.0] * 4096 for _ in range(4)]
        self.w = 0
        self.lp = [0.0] * 4
        self.ap_x1 = [0.0] * 4
        self.ap_y1 = [0.0] * 4
        self.hp_x1 = self.hp_y1 = 0.0
        self.dc_x1 = [0.0, 0.0]
        self.dc_y1 = [0.0, 0.0]
        self.delay = []              # smoothed delay lengths (glide on TUNE/CHRD changes)
        self.update()

    def set(self, **knobs):
        self.k.update(knobs)
        self.update()

    def update(self):
        k = self.k
        root = TUNE_LOW + k["TUNE"]
        self.notes = chord_notes(root, k["CHRD"], k["VOIC"], (k["INT2"], k["INT3"], k["INT4"]))
        frozen = bool(k["FREZ"])
        self.a = 0.0 if frozen else damping_coef(k["BRIT"])
        t60 = decay_seconds(k["DCAY"])
        target, self.g, self.inj = [], [], []
        a = self.a
        for n in self.notes:
            f0 = midi_hz(n)
            period = FS / f0
            w = 2.0 * math.pi * f0 / FS
            # the one-pole's phase delay at f0, so the loop rings exactly at the note
            lp_phase = math.atan2(a * math.sin(w), 1.0 - a * math.cos(w)) / w
            target.append(max(2.0, period - lp_phase))
            if frozen:
                g = 1.0
            else:
                lp_gain = (1.0 - a) / math.sqrt(1.0 - 2.0 * a * math.cos(w) + a * a)
                g = min(0.99995, 10.0 ** (-3.0 * period / (FS * t60)) / lp_gain)
            self.g.append(g)
            g_eff = min(1.0, g)
            self.inj.append(0.0 if frozen else math.sqrt(max(0.0, 1.0 - g_eff * g_eff)))
        self.target = target
        if not self.delay:
            self.delay = list(target)
        # stereo positions: SPRD 0 = all centre, 127 = -1, -1/3, +1/3, +1
        s = k["SPRD"] / 127.0
        self.pan = [s * p for p in (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)]
        self.drive = 1.0 + 7.0 * (k["DRIV"] / 127.0) ** 2
        self.mix = k["MIX"] / 127.0

    @staticmethod
    def soft(x):
        """Exactly linear below 0.5, a quadratic knee to a ceiling of 1.0 at 1.5.
        Linear below the knee is what lets FREZ hold forever; the knee bounds DRIV and FREZ.
        Polynomial only (no division), so the DSP computes the same curve in a few instructions."""
        ax = abs(x)
        if ax <= 0.5:
            return x
        u = min(ax - 0.5, 1.0)
        y = 0.5 + u - 0.5 * u * u
        return y if x > 0 else -y

    def process(self, l, r):
        x = 0.5 * (l + r)
        # fixed low cut, one-pole high pass at 60 Hz
        c = math.exp(-2.0 * math.pi * 60.0 / FS)
        hp = c * (self.hp_y1 + x - self.hp_x1)
        self.hp_x1, self.hp_y1 = x, hp
        outl = outr = 0.0
        w = self.w
        for i in range(4):
            # glide each delay toward its target (about 5 ms), so TUNE/CHRD/LFO moves sweep instead of click
            self.delay[i] += (self.target[i] - self.delay[i]) * 0.005
            d = self.delay[i]
            line = self.lines[i]
            # integer part N, fractional part in [0.1, 1.1) through a first-order allpass
            n_int = int(d - 0.1)
            frac = d - n_int
            c = (1.0 - frac) / (1.0 + frac)
            xn = line[(w - n_int) & 4095]
            y = c * xn + self.ap_x1[i] - c * self.ap_y1[i]
            self.ap_x1[i], self.ap_y1[i] = xn, y
            self.lp[i] = (1.0 - self.a) * y + self.a * self.lp[i]
            fb = self.g[i] * self.lp[i] + self.inj[i] * hp
            fb = self.soft(fb * self.drive) / self.drive
            line[w & 4095] = fb
            p = self.pan[i]
            outl += y * (1.0 - p) * 0.5
            outr += y * (1.0 + p) * 0.5
        self.w = (w + 1) & 4095
        wet = []
        for ch, v in enumerate((outl, outr)):
            v *= 0.5                                     # four voices, level with the input
            y = v - self.dc_x1[ch] + 0.9995 * self.dc_y1[ch]
            self.dc_x1[ch], self.dc_y1[ch] = v, y
            wet.append(y)
        m = self.mix
        return l + (wet[0] - l) * m, r + (wet[1] - r) * m


# ---------------------------------------------------------------- signals


def hits(seconds, bpm=124.0, every=2, seed=1):
    """Short noisy clicks on every second 16th: a stand-in for a rim / hat track."""
    rnd = random.Random(seed)
    n = int(seconds * FS)
    step = int(FS * 60.0 / bpm / 4.0)
    out = [0.0] * n
    for s in range(0, n, step * every):
        for j in range(220):
            if s + j < n:
                out[s + j] = (rnd.uniform(-1, 1) * 0.6 + (0.8 if j < 3 else 0.0)) * math.exp(-j / 40.0)
    return out


def burst(seconds, length=0.03, seed=2):
    rnd = random.Random(seed)
    n = int(seconds * FS)
    m = int(length * FS)
    return [rnd.uniform(-0.7, 0.7) if i < m else 0.0 for i in range(n)]


def run(res, mono, changes=None):
    """Render mono input through one instance. changes: {sample: {knob: value}}."""
    L, R = [], []
    for i, x in enumerate(mono):
        if changes and i in changes:
            res.set(**changes[i])
        l, r = res.process(x, x)
        L.append(l)
        R.append(r)
    return L, R


def write_wav(path, L, R):
    peak = max(1e-9, max(max(abs(v) for v in L), max(abs(v) for v in R)))
    scale = 0.89 / peak                                # demos: peak-normalized to -1 dBFS
    with wave.open(path, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(int(FS))
        f.writeframes(b"".join(struct.pack("<hh", int(l * scale * 32767), int(r * scale * 32767))
                               for l, r in zip(L, R)))
    return peak


# ---------------------------------------------------------------- checks


def goertzel_power(x, f):
    w = 2.0 * math.pi * f / FS
    c = 2.0 * math.cos(w)
    s1 = s2 = 0.0
    for v in x:
        s1, s2 = v + c * s1 - s2, s1
    return s1 * s1 + s2 * s2 - c * s1 * s2


def measured_pitch(x, f0):
    """Peak frequency near f0, found by a Goertzel scan in 1-cent steps over +-60 cents."""
    best = max(range(-60, 61), key=lambda c: goertzel_power(x, f0 * 2 ** (c / 1200.0)))
    return best


def single_voice_ir(note, brit=127, dcay=100, seconds=0.6):
    """Impulse response of one comb tuned to `note` (USER chord, all notes the same)."""
    root = note - TUNE_LOW
    r = Resonator(TUNE=root, CHRD=0, VOIC=0, INT2=24, INT3=24, INT4=24, BRIT=brit, DCAY=dcay,
                  MIX=127, SPRD=0)
    x = [0.0] * int(seconds * FS)
    x[0] = 1.0
    L, _ = run(r, x)
    return L[2000:]                                   # skip the onset


def test():
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok = ok and cond

    # 1. tuning: each comb rings at its note, within 5 cents, open and damped
    for note in (36, 45, 57, 69, 81):
        for brit in (127, 60):
            if TUNE_LOW <= note < TUNE_LOW + TUNE_COUNT:
                ir = single_voice_ir(note, brit=brit)
                cents = measured_pitch(ir, midi_hz(note))
                check(abs(cents) <= 5, f"tuning note {note} BRIT {brit}: {cents:+d} cents")

    # 2. chords and voicings give the intended notes
    check(chord_notes(48, 11, 0) == [48, 55, 58, 63], "DUB on C3 = C3 G3 Bb3 Eb4")
    check(chord_notes(48, 3, 1) == [51, 55, 60, 60], "MIN INV1 on C3 = Eb3 G3 C4 C4")
    check(chord_notes(48, 0, 0, (24 + 5, 24 + 10, 24 - 12)) == [36, 48, 53, 58], "USER -12/+5/+10 on C3")
    check(all(LOWEST_NOTE <= n <= HIGHEST_NOTE for c in range(len(CHORDS)) for v in range(len(VOICINGS))
              for t in (0, TUNE_COUNT - 1) for n in chord_notes(TUNE_LOW + t, c, v)),
          "every chord x voicing x TUNE end stays inside C1..C8")

    # 3. equal decay: low and high voices lose 60 dB in about the same time
    def t60_measured(note):
        ir = single_voice_ir(note, brit=127, dcay=70, seconds=3.0)
        blk = 4096                                     # longer than any period
        env = [math.sqrt(sum(v * v for v in ir[i:i + blk]) / blk) for i in range(0, len(ir) - blk, blk // 4)]
        top = env[0]
        t10 = t40 = None
        for i, e in enumerate(env):
            if t10 is None and e < top * 10 ** (-10 / 20):
                t10 = i * (blk // 4) / FS
            if t40 is None and e < top * 10 ** (-40 / 20):
                t40 = i * (blk // 4) / FS
        if t10 is None or t40 is None:
            return float("inf")
        return (t40 - t10) * 2.0                        # 30 dB of slope -> 60 dB

    lo, hi = t60_measured(40), t60_measured(76)
    want = decay_seconds(70)
    check(abs(lo - hi) / want < 0.25, f"equal decay: low {lo:.2f}s high {hi:.2f}s (DCAY 70 = {want:.2f}s)")

    # 4. stability: max DCAY, max DRIV, a hot noise input -> bounded, and decays after
    r = Resonator(DCAY=127, DRIV=127, BRIT=127, MIX=127)
    x = burst(1.5, length=0.5)
    L, R = run(r, [v * 1.5 for v in x])
    pk = max(max(abs(v) for v in L), max(abs(v) for v in R))
    check(pk < 2.0, f"bounded at max DCAY/DRIV on a hot input: peak {pk:.2f}")

    # 5. FREZ holds, and releases
    r = Resonator(DCAY=80, MIX=127)
    x = burst(4.0)
    L, _ = run(r, x, changes={int(0.2 * FS): {"FREZ": 1}, int(2.5 * FS): {"FREZ": 0}})
    def rms(a, b):
        seg = L[int(a * FS):int(b * FS)]
        return math.sqrt(sum(v * v for v in seg) / len(seg))
    held_a, held_b, after = rms(0.5, 0.7), rms(2.2, 2.4), rms(3.8, 4.0)
    check(held_b > held_a * 0.5, f"FREZ holds: {20*math.log10(held_b/held_a):+.1f} dB over 1.7 s")
    check(after < held_b * 0.1, f"FREZ off decays: {20*math.log10(max(after,1e-12)/held_b):+.1f} dB after 1.4 s")

    # 6. MIX 0 is an exact passthrough
    r = Resonator(MIX=0)
    x = hits(0.5)
    L, R = run(r, x)
    check(all(a == b for a, b in zip(L, x)) and all(a == b for a, b in zip(R, x)), "MIX 0 is bit-exact dry")

    # 7. silence after the input stops (no limit cycle / DC)
    r = Resonator(DCAY=60, MIX=127)
    x = burst(6.0)
    L, R = run(r, x)
    tail = max(max(abs(v) for v in L[-int(0.5 * FS):]), max(abs(v) for v in R[-int(0.5 * FS):]))
    check(tail < 10 ** (-90 / 20), f"settles after the input stops: {20*math.log10(max(tail,1e-12)):.0f} dBFS")

    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


def render(outdir):
    import os
    os.makedirs(outdir, exist_ok=True)
    demos = [
        ("01_dub_stab_hits", dict(TUNE=24 - 12 + 2, CHRD=11, VOIC=0, DCAY=85, BRIT=70, SPRD=90, MIX=110), hits(8.0), None),
        ("02_dub_minor9_open", dict(TUNE=14, CHRD=8, VOIC=5, DCAY=100, BRIT=55, SPRD=127, MIX=100), hits(8.0, every=3), None),
        ("03_chord_walk_lfo_like", dict(TUNE=14, CHRD=5, VOIC=0, DCAY=80, BRIT=75, SPRD=100, MIX=110), hits(8.0),
         {int(2 * FS): {"CHRD": 8}, int(4 * FS): {"CHRD": 11, "VOIC": 2}, int(6 * FS): {"CHRD": 13}}),
        ("04_industrial_cluster_drive", dict(TUNE=10, CHRD=14, VOIC=6, DCAY=95, BRIT=110, SPRD=127, DRIV=110, MIX=120), hits(8.0), None),
        ("05_freeze_drone", dict(TUNE=14, CHRD=12, VOIC=5, DCAY=100, BRIT=60, SPRD=127, MIX=127), burst(8.0, 0.08),
         {int(0.3 * FS): {"FREZ": 1}, int(6.5 * FS): {"FREZ": 0}}),
        ("06_user_chord_tuned", dict(TUNE=14, CHRD=0, INT2=24 + 7, INT3=24 + 10, INT4=24 + 17, VOIC=0, DCAY=90, BRIT=80, SPRD=100, MIX=110), hits(8.0), None),
    ]
    for name, knobs, sig, changes in demos:
        r = Resonator(**knobs)
        L, R = run(r, sig, changes)
        path = f"{outdir}/{name}.wav"
        pk = write_wav(path, L, R)
        print(f"{path}  peak {20*math.log10(pk):+.1f} dBFS  notes {r.notes}")
    dry = hits(8.0)
    write_wav(f"{outdir}/00_dry_input.wav", dry, dry)
    print(f"{outdir}/00_dry_input.wav")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "test":
        sys.exit(test())
    if len(sys.argv) >= 3 and sys.argv[1] == "render":
        render(sys.argv[2])
        sys.exit(0)
    print(__doc__)
