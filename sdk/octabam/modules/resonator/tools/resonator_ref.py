#!/usr/bin/env python3
"""RESONATOR reference model v0.2: the arithmetic the DSP56300 engine computes.

A chord resonator insert for the Octatrack's FX2 slot. The track's audio (mono
sum, a 60 Hz one-pole low cut) excites four tuned feedback combs whose pitches
form a chord:

    in -> HP 60 Hz -> 4 x [ ring -> allpass (fraction) -> FIR damping -> g ] -> S/D pan -> makeup -> MIX

Per comb, per sample (all four rings advance together, one write per sample):
    xn   = ring[w - N]
    y    = c*xn + x1 - c*y1                 first-order allpass: the fractional delay
    filt = y1 + b*(y + y2 - 2*y1)           3-tap FIR [b, 1-2b, b]: brightness, 1 sample delay
    ring[w] = clip(g*filt + injD*h)         the DSP's store limiter is the clip (+-1)
Loop delay = N + frac + 1 = P, the note's period, so every note is in tune.

Block-rate (once per call, from the knobs):
    note  = 36 + TUNE + chord offset + voicing octave, folded into C1..C8
    U_P   = 11 - log2(P)       -> P/2048 = 2^-U_P
    U_x   = U_P + K_X + K_D*DCAY   -> x/16 = 2^-U_x  (x = P*r, r per-sample log2 loss)
    g     = 2^-x               -> every note rings for the DCAY time (T60 50 ms .. 20 s)
    inj   = 2^-((U_x - K_I)/2), at most 1   -> ~sqrt(1-g^2): energy-normalized input
    injD  = inj * D/4, D = 2^(2*DRIV/127) (1..4): DRIV pushes the rings into the clip
FREZ: g = 1, b = 0, inj = 0: the ring holds exactly (the loop is lossless and linear).

Controls (12 slots):
  page 1  TUNE DCAY BRIT SPRD DRIV MIX
  page 2  CHRD VOIC INT2 INT3 INT4 FREZ    (slot pairs share x:(r6+$c/$d/$e): knob | companion)

Original design from published techniques (feedback comb / Karplus-Strong
resonators, Thiran allpass interpolation, Jot-style T60 loop gain). No code,
tables or sounds from any instrument's firmware.

    python3 resonator_ref.py test               checks
    python3 resonator_ref.py render OUTDIR      WAV demos
    python3 resonator_ref.py tables             the ptable words and constants for the DSP
"""
import math
import random
import struct
import sys
import wave

FS = 44100.0
BLOCK = 16

TUNE_LOW = 36            # TUNE 0 = C2
TUNE_COUNT = 60          # C2..B6
LOWEST_NOTE = 24         # voices fold up by octaves until >= C1 (P = 1349 samples)
HIGHEST_NOTE = 108       # and down until <= C8
RING = 2048              # words per comb ring (FX2 buffer: 4 rings = 8192 words)

CHORDS = [               # order is API once released (projects store the index)
    ("USER", None),
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
    ("DUB",  (0, 7, 10, 15)),
    ("MI11", (0, 7, 10, 17)),
    ("QRTL", (0, 5, 10, 15)),
    ("CLST", (0, 1, 6, 7)),
]
# VOIC: octave offsets per chord SLOT. For the preset chords (ascending) these are
# the usual voicings: INVn = lowest n notes up an octave, DRP2 = second-highest down,
# OPEN = 2nd and 4th up, WIDE = root down, 3rd and 4th up. USER chords use the same slots.
VOICINGS = [
    ("CLSE", (0, 0, 0, 0)),
    ("INV1", (12, 0, 0, 0)),
    ("INV2", (12, 12, 0, 0)),
    ("INV3", (12, 12, 12, 0)),
    ("DRP2", (0, 0, -12, 0)),
    ("OPEN", (0, 12, 0, 12)),
    ("WIDE", (-12, 0, 12, 12)),
]
PAN_W = (-1.0, -0.5, 0.5, 1.0)          # voice positions at SPRD 127
MAKEUP = 4.0                            # wet makeup (asl #2)
LOG2_FS_A4 = math.log2(FS / 440.0)
# x = P * r with r = 3*log2(10) / (FS * T60) and T60 = 0.05 * 400^(DCAY/127):
# x/16 = 2^-(U_P + K_X + K_D*DCAY)  with  K_X = -log2(2048 * 3*log2(10) / (FS*0.05) / 16)
K_X = -math.log2(2048.0 * 3.0 * math.log2(10.0) / (FS * 0.05) / 16.0)
K_D = math.log2(400.0) / 127.0
# inj = sqrt(2 ln2 x) = 2^-((U_x - K_I)/2) with x = 16 * 2^-U_x
K_I = math.log2(16.0 * 2.0 * math.log(2.0))
C_HP = math.exp(-2.0 * math.pi * 60.0 / FS)


def note_offsets(chord, voicing, ints):
    shape = CHORDS[chord][1]
    if shape is None:
        shape = (0,) + tuple(i - 24 for i in ints)
    return [s + o for s, o in zip(shape, VOICINGS[voicing][1])]


def fold(n):
    for _ in range(3):
        if n < LOWEST_NOTE:
            n += 12
    for _ in range(3):
        if n > HIGHEST_NOTE:
            n -= 12
    return n


def chord_notes(tune, chord, voicing, ints=(27, 31, 34)):
    return [fold(TUNE_LOW + tune + o) for o in note_offsets(chord, voicing, ints)]


def u_p(note):
    """11 - log2(period): P/2048 = 2^-U_P."""
    return 11.0 - (LOG2_FS_A4 + (69 - note) / 12.0)


def decay_seconds(dcay):
    return 0.05 * (400.0 ** (dcay / 127.0))


def coefficients(k):
    """Block-rate values for one call, exactly as the DSP derives them (in float)."""
    frozen = (k["FREZ"] & 0x7F) != 0
    notes = chord_notes(k["TUNE"], k["CHRD"], k["VOIC"], (k["INT2"], k["INT3"], k["INT4"]))
    drive_q = 2.0 ** (2.0 * k["DRIV"] / 127.0 - 2.0)                 # D/4
    out = []
    for n in notes:
        up = u_p(n)
        p = 2048.0 * 2.0 ** (-up)
        ux = min(up + K_X + K_D * k["DCAY"], 16.0)
        x = 16.0 * 2.0 ** (-ux)
        g = 1.0 if frozen else 2.0 ** (-x)
        ui = (ux - K_I) / 2.0
        inj = 0.0 if frozen else min(1.0, 2.0 ** (-ui))
        out.append(dict(note=n, P=p, g=g, injD=inj * drive_q))
    b = 0.0 if frozen else 0.25 * (1.0 - k["BRIT"] / 128.0)
    s = k["SPRD"] / 128.0
    mix = 1.0 if k["MIX"] >= 127 else k["MIX"] / 128.0
    return out, b, s, mix


# ---------------------------------------------------------------- engine


class Resonator:
    def __init__(self, **knobs):
        self.k = dict(TUNE=14, DCAY=90, BRIT=80, SPRD=64, DRIV=0, MIX=64,
                      CHRD=11, VOIC=0, INT2=27, INT3=31, INT4=34, FREZ=0)
        self.k.update(knobs)
        self.rings = [[0.0] * RING for _ in range(4)]
        self.w = 0
        self.apx = [0.0] * 4
        self.apy = [0.0] * 4
        self.z2 = [0.0] * 4
        self.hm = self.hh = 0.0
        self.ds = [-1.0] * 4             # smoothed delay; negative = snap to the target

    def set(self, **knobs):
        self.k.update(knobs)

    def block(self, frames):
        """frames: list of (l, r). Returns the processed list."""
        voices, b, s, mix = coefficients(self.k)
        setup = []
        for v, c in enumerate(voices):
            target = c["P"] - 1.0
            if self.ds[v] < 0.0:
                self.ds[v] = target
            else:
                self.ds[v] += (target - self.ds[v]) / 8.0
            d = self.ds[v]
            n_int = math.floor(d - 0.5)
            frac = d - n_int
            ap = (1.0 - frac) / (1.0 + frac)
            setup.append((n_int, ap, c["g"], c["injD"]))
        out = []
        for l, r in frames:
            m = 0.5 * (l + r)
            h = C_HP * (self.hh + m - self.hm)
            self.hm, self.hh = m, h
            S = D = 0.0
            for v, (n_int, ap, g, inj) in enumerate(setup):
                ring = self.rings[v]
                xn = ring[(self.w - n_int) % RING]
                y = ap * xn + self.apx[v] - ap * self.apy[v]
                y1 = self.apy[v]
                self.apx[v] = xn
                filt = y1 + b * (y + self.z2[v] - 2.0 * y1)
                self.z2[v], self.apy[v] = y1, y
                ring[self.w] = max(-1.0, min(1.0, g * filt + inj * h))
                S += y / 4.0
                D += PAN_W[v] * y / 4.0
            self.w = (self.w + 1) % RING
            wl = (S - s * D) * MAKEUP
            wr = (S + s * D) * MAKEUP
            out.append((l + (wl - l) * mix, r + (wr - r) * mix))
        return out

    @property
    def notes(self):
        return chord_notes(self.k["TUNE"], self.k["CHRD"], self.k["VOIC"],
                           (self.k["INT2"], self.k["INT3"], self.k["INT4"]))


def run(res, mono, changes=None):
    """Render mono input in 16-frame blocks. changes: {sample: {knob: value}}, applied at block starts."""
    L, R = [], []
    for i in range(0, len(mono), BLOCK):
        if changes:
            for at, kn in changes.items():
                if i <= at < i + BLOCK:
                    res.set(**kn)
        for l, r in res.block([(x, x) for x in mono[i:i + BLOCK]]):
            L.append(l)
            R.append(r)
    return L, R


# ---------------------------------------------------------------- signals


def hits(seconds, bpm=124.0, every=2, seed=1, level=0.9):
    rnd = random.Random(seed)
    n = int(seconds * FS)
    step = int(FS * 60.0 / bpm / 4.0)
    out = [0.0] * n
    for s in range(0, n, step * every):
        for j in range(220):
            if s + j < n:
                out[s + j] = level * (rnd.uniform(-1, 1) * 0.6 + (0.8 if j < 3 else 0.0)) * math.exp(-j / 40.0)
    return out


def burst(seconds, length=0.03, seed=2, level=0.7):
    rnd = random.Random(seed)
    n = int(seconds * FS)
    m = int(length * FS)
    return [rnd.uniform(-level, level) if i < m else 0.0 for i in range(n)]


def write_wav(path, L, R, normalize=False):
    peak = max(1e-9, max(max(abs(v) for v in L), max(abs(v) for v in R)))
    scale = 0.89 / peak if normalize else 1.0
    with wave.open(path, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(int(FS))
        f.writeframes(b"".join(struct.pack("<hh", max(-32768, min(32767, int(l * scale * 32767))),
                                           max(-32768, min(32767, int(r * scale * 32767))))
                               for l, r in zip(L, R)))
    return peak


# ---------------------------------------------------------------- tables for the DSP


def q23(v):
    w = int(round(v * (1 << 23)))
    w = max(-(1 << 23), min((1 << 23) - 1, w))
    return w & 0xFFFFFF


def ap_table():
    """c = (1-f)/(1+f) for f = 0.5 + i/64, i = 0..64 (65 words, linear interpolation)."""
    return [q23((1.0 - f) / (1.0 + f)) for f in (0.5 + i / 64.0 for i in range(65))]


def ptable():
    words = list(ap_table())
    for _, shape in CHORDS:
        words += [o & 0xFFFFFF for o in (shape or (0, 0, 0, 0))]
    for _, octs in VOICINGS:
        words += [o & 0xFFFFFF for o in octs]
    return words


def exp2_poly():
    """Least-squares cubic for 0.5 * 2^-f on [0,1), and its worst relative error."""
    n = 400
    xs = [i / (n - 1) for i in range(n)]
    ys = [0.5 * 2.0 ** (-x) for x in xs]
    A = [[sum(x ** (i + j) for x in xs) for j in range(4)] for i in range(4)]
    B = [sum(y * x ** i for x, y in zip(xs, ys)) for i in range(4)]
    for i in range(4):                                    # Gauss-Jordan
        p = A[i][i]
        A[i] = [a / p for a in A[i]]
        B[i] /= p
        for j in range(4):
            if j != i:
                f = A[j][i]
                A[j] = [a - f * b for a, b in zip(A[j], A[i])]
                B[j] -= f * B[i]
    err = max(abs((B[0] + x * (B[1] + x * (B[2] + x * B[3]))) / (0.5 * 2.0 ** -x) - 1.0) for x in xs)
    return B, err


def tables():
    t = ptable()
    print(f"ptable: {len(t)} words (AP 65 @0, CHORD 60 @65, VOIC 28 @125)")
    coef, err = exp2_poly()
    print("exp2 poly 0.5*2^-f:", ", ".join(f"${q23(c):06x} ({c:+.8f})" for c in coef),
          f"max rel err {err:.2e} = {1200*math.log2(1+err):.3f} cents")
    k0 = (11.0 - LOG2_FS_A4 - 69 / 12.0) / 16.0
    print(f"U_P/16 = (note/128)*(2/3) + {k0:+.8f} -> ${q23(k0):06x}; 2/3 = ${q23(2/3):06x}")
    print(f"K_X/16 = {K_X/16:+.8f} -> ${q23(K_X/16):06x};  K_D*128/16 = {K_D*128/16:.8f} -> ${q23(K_D*128/16):06x}")
    print(f"K_I/16 = {K_I/16:.8f} -> ${q23(K_I/16):06x}")
    print(f"C_HP = {C_HP:.8f} -> ${q23(C_HP):06x}")
    print(f"drive U_D/16 = 0.125 - (knob/128)*{2*128/127/16:.8f} -> ${q23(2*128/127/16):06x}")
    return t


# ---------------------------------------------------------------- checks


def goertzel_power(x, f):
    w = 2.0 * math.pi * f / FS
    c = 2.0 * math.cos(w)
    s1 = s2 = 0.0
    for v in x:
        s1, s2 = v + c * s1 - s2, s1
    return s1 * s1 + s2 * s2 - c * s1 * s2


def measured_pitch(x, f0):
    return max(range(-60, 61), key=lambda c: goertzel_power(x, f0 * 2 ** (c / 1200.0)))


def single_voice_ir(note, brit=127, dcay=100, seconds=0.6):
    r = Resonator(TUNE=note - TUNE_LOW, CHRD=0, VOIC=0, INT2=24, INT3=24, INT4=24, BRIT=brit, DCAY=dcay,
                  MIX=127, SPRD=0)
    x = [0.0] * int(seconds * FS)
    x[0] = 1.0
    L, _ = run(r, x)
    return L[2000:]


def test():
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok = ok and cond

    _, err = exp2_poly()
    check(1200 * math.log2(1 + err) < 0.5, f"exp2 cubic within 0.5 cent ({1200*math.log2(1+err):.3f})")
    check(len(ptable()) == 65 + 60 + 28, "ptable 153 words")

    for note in (36, 45, 57, 69, 81):
        for brit in (127, 40):
            cents = measured_pitch(single_voice_ir(note, brit=brit), 440.0 * 2 ** ((note - 69) / 12))
            check(abs(cents) <= 3, f"tuning note {note} BRIT {brit}: {cents:+d} cents")

    check(chord_notes(12, 11, 0) == [48, 55, 58, 63], "DUB on C3 = C3 G3 Bb3 Eb4")
    check(sorted(chord_notes(12, 3, 1)) == [51, 55, 60, 60], "MIN INV1 on C3 = Eb3 G3 C4 C4")
    check(sorted(chord_notes(12, 0, 0, (29, 34, 12))) == [36, 48, 53, 58], "USER -12/+5/+10 on C3")
    check(all(LOWEST_NOTE <= n <= HIGHEST_NOTE for c in range(len(CHORDS)) for v in range(len(VOICINGS))
              for t in (0, TUNE_COUNT - 1) for i in ((0, 0, 0), (48, 48, 48))
              for n in chord_notes(t, c, v, i)),
          "every chord x voicing x TUNE x INT end stays inside C1..C8")
    check(2048 * 2 ** -u_p(LOWEST_NOTE) < RING - 4, "longest period fits a 2048-word ring")

    def t60_measured(note):
        ir = single_voice_ir(note, brit=127, dcay=70, seconds=3.0)
        blk = 4096
        env = [math.sqrt(sum(v * v for v in ir[i:i + blk]) / blk) for i in range(0, len(ir) - blk, blk // 4)]
        top, t10, t40 = env[0], None, None
        for i, e in enumerate(env):
            if t10 is None and e < top * 10 ** (-10 / 20):
                t10 = i * (blk // 4) / FS
            if t40 is None and e < top * 10 ** (-40 / 20):
                t40 = i * (blk // 4) / FS
        return float("inf") if t10 is None or t40 is None else (t40 - t10) * 2.0

    lo, hi, want = t60_measured(40), t60_measured(76), decay_seconds(70)
    check(abs(lo - want) / want < 0.2 and abs(hi - want) / want < 0.2,
          f"decay time: low {lo:.2f}s high {hi:.2f}s (DCAY 70 = {want:.2f}s)")

    r = Resonator(DCAY=127, DRIV=127, BRIT=127, MIX=127)
    L, R = run(r, [v * 1.4 for v in burst(1.0, length=0.5)])
    pk = max(max(abs(v) for v in L), max(abs(v) for v in R))
    check(pk <= 4 * 1.75 + 1e-9, f"bounded at max DCAY/DRIV on a hot input: wet peak {pk:.2f}")

    r = Resonator(DCAY=80, MIX=127)
    L, _ = run(r, burst(4.0), changes={int(0.2 * FS): {"FREZ": 1}, int(2.5 * FS): {"FREZ": 0}})

    def rms(a, b):
        seg = L[int(a * FS):int(b * FS)]
        return math.sqrt(sum(v * v for v in seg) / len(seg))

    held_a, held_b, after = rms(0.5, 0.7), rms(2.2, 2.4), rms(3.8, 4.0)
    check(abs(20 * math.log10(held_b / held_a)) < 0.5, f"FREZ holds: {20*math.log10(held_b/held_a):+.2f} dB over 1.7 s")
    check(after < held_b * 0.1, f"FREZ off decays: {20*math.log10(max(after,1e-12)/held_b):+.1f} dB after 1.4 s")

    r = Resonator(MIX=0)
    x = hits(0.5)
    L, R = run(r, x)
    check(all(a == b for a, b in zip(L, x)) and all(a == b for a, b in zip(R, x)), "MIX 0 is bit-exact dry")

    r = Resonator(DCAY=60, MIX=127)
    L, R = run(r, burst(6.0))
    tail = max(max(abs(v) for v in L[-int(0.5 * FS):]), max(abs(v) for v in R[-int(0.5 * FS):]))
    check(tail < 10 ** (-90 / 20), f"settles after the input stops: {20*math.log10(max(tail,1e-12)):.0f} dBFS")

    r = Resonator(TUNE=14, CHRD=11, DCAY=85, BRIT=70, SPRD=90, MIX=127)
    L, R = run(r, hits(4.0))
    pk = max(max(abs(v) for v in L), max(abs(v) for v in R))
    check(10 ** (-18 / 20) < pk < 1.0, f"dub stab wet level at MIX 127: {20*math.log10(pk):+.1f} dBFS")

    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


def demos():
    return [
        ("01_dub_stab_hits", dict(TUNE=14, CHRD=11, VOIC=0, DCAY=85, BRIT=70, SPRD=90, MIX=110), hits(8.0), None),
        ("02_dub_minor9_open", dict(TUNE=14, CHRD=8, VOIC=5, DCAY=100, BRIT=55, SPRD=127, MIX=100), hits(8.0, every=3), None),
        ("03_chord_walk", dict(TUNE=14, CHRD=5, VOIC=0, DCAY=80, BRIT=75, SPRD=100, MIX=110), hits(8.0),
         {int(2 * FS): {"CHRD": 8}, int(4 * FS): {"CHRD": 11, "VOIC": 2}, int(6 * FS): {"CHRD": 13}}),
        ("04_industrial_cluster_drive", dict(TUNE=10, CHRD=14, VOIC=6, DCAY=95, BRIT=110, SPRD=127, DRIV=110, MIX=120), hits(8.0), None),
        ("05_freeze_drone", dict(TUNE=14, CHRD=12, VOIC=5, DCAY=100, BRIT=60, SPRD=127, MIX=127), burst(8.0, 0.08),
         {int(0.3 * FS): {"FREZ": 1}, int(6.5 * FS): {"FREZ": 0}}),
        ("06_user_chord", dict(TUNE=14, CHRD=0, INT2=24 + 7, INT3=24 + 10, INT4=24 + 17, VOIC=0, DCAY=90, BRIT=80, SPRD=100, MIX=110), hits(8.0), None),
    ]


def render(outdir):
    import os
    os.makedirs(outdir, exist_ok=True)
    for name, knobs, sig, changes in demos():
        r = Resonator(**knobs)
        L, R = run(r, sig, changes)
        pk = write_wav(f"{outdir}/{name}.wav", L, R)
        print(f"{outdir}/{name}.wav  peak {20*math.log10(pk):+.1f} dBFS  notes {r.notes}")
    dry = hits(8.0)
    write_wav(f"{outdir}/00_dry_input.wav", dry, dry)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "test":
        sys.exit(test())
    if cmd == "render" and len(sys.argv) > 2:
        render(sys.argv[2])
        sys.exit(0)
    if cmd == "tables":
        tables()
        sys.exit(0)
    print(__doc__)
