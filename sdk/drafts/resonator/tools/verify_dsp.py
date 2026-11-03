#!/usr/bin/env python3
"""Render cases through the assembled engine (emulator, no firmware) and check them.

    python3 tools/verify_dsp.py [--keep DIR]

Each case is rendered twice: by resonator_ref.py (float) and by resonator.asm in
dsp56kEmu through tools/harness (Docker image modwerk-source-tools). The DSP run
starts with 64 dry clearing calls (1,024 frames), which are skipped here.

Checks, all on the DSP output:
  * tuning of single voices against the note (Goertzel, 1-cent steps)
  * the decay time against DCAY
  * FREZ holds and releases; MIX 0 is bit-exact dry; silence after the input stops
  * no Y write outside the four rings (the harness fills the rest with garbage)
  * agreement with the model (SNR over the first second of each case)
  * cost: the most instructions one 16-frame call took
"""
import math
import os
import pathlib
import struct
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
MOD = HERE.parent
sys.path.insert(0, str(HERE))
import resonator_ref as ref  # noqa: E402

FS = ref.FS
CLEAR = 64 * 16
_N = 0
SLOTS = ["TUNE", "DCAY", "BRIT", "SPRD", "DRIV", "MIX", "CHRD", "VOIC", "INT2", "INT3", "INT4", "FREZ"]


def write_f32(path, xs):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(xs)}f", *xs))


def read_f32(path):
    data = pathlib.Path(path).read_bytes()
    return list(struct.unpack(f"<{len(data)//4}f", data))


def dsp_render(knobs, mono, changes, tmp):
    base = dict(TUNE=14, DCAY=90, BRIT=80, SPRD=64, DRIV=0, MIX=64,
                CHRD=11, VOIC=0, INT2=27, INT3=31, INT4=34, FREZ=0)
    base.update(knobs)
    lines = [f"0 {i} {base[s]}" for i, s in enumerate(SLOTS)]
    for at, kn in (changes or {}).items():
        blk = (at // 16) * 16
        for s, v in kn.items():
            lines.append(f"{blk + CLEAR} {SLOTS.index(s)} {v}")
    global _N
    _N += 1
    tag = f"c{_N}"
    ev = pathlib.Path(tmp) / f"{tag}.ev"
    ev.write_text("\n".join(lines) + "\n")
    write_f32(pathlib.Path(tmp) / f"{tag}.in", [0.0] * CLEAR + mono)
    (MOD / "tools/harness/.build").mkdir(parents=True, exist_ok=True)
    (MOD / "tools/harness/.build/ptable.txt").write_text("\n".join(f"{w:06x}" for w in ref.ptable()) + "\n")
    p = subprocess.run(["docker", "run", "--rm", "-v", f"{MOD}:/w", "-v", f"{tmp}:/t", "modwerk-source-tools",
                        "sh", "/w/tools/harness/run.sh", f"/t/{tag}.in", f"/t/{tag}.out", f"/t/{tag}.ev"],
                       capture_output=True, text=True)
    if p.returncode not in (0, 3):
        raise SystemExit(f"harness failed ({p.returncode}):\n{p.stdout}\n{p.stderr}")
    out = read_f32(pathlib.Path(tmp) / f"{tag}.out")
    if len(out) != 2 * (CLEAR + len(mono)):
        raise SystemExit(f"harness wrote {len(out)//2} frames for {CLEAR + len(mono)}")
    L, R = out[0::2][CLEAR:], out[1::2][CLEAR:]
    return L, R, p.stdout.strip(), p.returncode


def snr(a, b):
    num = sum(x * x for x in a) or 1e-30
    den = sum((x - y) ** 2 for x, y in zip(a, b)) or 1e-30
    return 10 * math.log10(num / den)


def main():
    keep = None
    if "--keep" in sys.argv:
        keep = sys.argv[sys.argv.index("--keep") + 1]
        os.makedirs(keep, exist_ok=True)
    ok = True
    worst = 0

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok = ok and cond

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        # tuning, single voices
        for note in (36, 57, 81):
            for brit in (127, 40):
                kn = dict(TUNE=note - ref.TUNE_LOW, CHRD=0, VOIC=0, INT2=24, INT3=24, INT4=24, BRIT=brit,
                          DCAY=100, MIX=127, SPRD=0)
                x = [0.0] * int(0.6 * FS)
                x[0] = 0.9
                L, _, info, rc = dsp_render(kn, x, None, tmp)
                cents = ref.measured_pitch(L[2000:], 440.0 * 2 ** ((note - 69) / 12))
                check(abs(cents) <= 3, f"DSP tuning note {note} BRIT {brit}: {cents:+d} cents")
                check(rc == 0, f"  no stray Y writes ({info.split('stray Y writes')[-1].strip()})")
                worst = max(worst, int(info.split("instructions max ")[1].split()[0]))

        # cases against the model
        for name, knobs, sig, changes in ref.demos():
            sig = sig[: int(3.0 * FS)]
            changes = {k: v for k, v in (changes or {}).items() if k < len(sig)}
            r = ref.Resonator(**knobs)
            rl, rr = ref.run(r, sig, changes)
            L, R, info, rc = dsp_render(knobs, sig, changes, tmp)
            worst = max(worst, int(info.split("instructions max ")[1].split()[0]))
            pk = max(max(abs(v) for v in L), max(abs(v) for v in R))
            if knobs.get("DRIV", 0) or changes:
                # clipping and a lossless loop diverge chaotically: compare loudness, 0.25 s windows
                w = int(0.25 * FS)
                env = lambda a: [math.sqrt(sum(v * v for v in a[i:i + w]) / w) + 1e-9 for i in range(0, len(a) - w, w)]
                dev = max(abs(20 * math.log10(p / q)) for p, q in zip(env(L), env(rl)))
                check(dev < 3.0 and rc == 0, f"{name}: DSP vs model loudness within {dev:.1f} dB (0.25 s windows), "
                                             f"DSP peak {20*math.log10(max(pk,1e-9)):+.1f} dBFS, stray-free {rc == 0}")
            else:
                s1 = snr(rl[: int(FS)], L[: int(FS)])
                check(s1 > 20 and rc == 0, f"{name}: DSP vs model {s1:.1f} dB SNR over 1 s, DSP peak "
                                           f"{20*math.log10(max(pk,1e-9)):+.1f} dBFS, stray-free {rc == 0}")
            if keep:
                ref.write_wav(f"{keep}/{name}_dsp.wav", L, R)
                ref.write_wav(f"{keep}/{name}_model.wav", rl, rr)

        # decay time
        kn = dict(TUNE=40 - ref.TUNE_LOW, CHRD=0, VOIC=0, INT2=24, INT3=24, INT4=24, BRIT=127, DCAY=70, MIX=127, SPRD=0)
        x = [0.0] * int(3.0 * FS)
        x[0] = 0.9
        L, _, _, _ = dsp_render(kn, x, None, tmp)
        blk = 4096
        env = [math.sqrt(sum(v * v for v in L[i:i + blk]) / blk) for i in range(2000, len(L) - blk, blk // 4)]
        top, t10, t40 = env[0], None, None
        for i, e in enumerate(env):
            if t10 is None and e < top * 10 ** (-10 / 20):
                t10 = i * (blk // 4) / FS
            if t40 is None and e < top * 10 ** (-40 / 20):
                t40 = i * (blk // 4) / FS
        t60 = float("inf") if t10 is None or t40 is None else (t40 - t10) * 2
        want = ref.decay_seconds(70)
        check(abs(t60 - want) / want < 0.2, f"DSP decay time {t60:.2f}s (DCAY 70 = {want:.2f}s)")

        # freeze
        # unison (four voices on one note), so no beating between voices moves the level
        L, _, _, _ = dsp_render(dict(DCAY=80, MIX=127, CHRD=0, INT2=24, INT3=24, INT4=24, SPRD=0), ref.burst(4.0),
                                {int(0.2 * FS): {"FREZ": 1}, int(2.5 * FS): {"FREZ": 0}}, tmp)

        def rms(a, b):
            seg = L[int(a * FS):int(b * FS)]
            return math.sqrt(sum(v * v for v in seg) / len(seg))

        ha, hb, af = rms(0.5, 0.7), rms(2.2, 2.4), rms(3.8, 4.0)
        check(abs(20 * math.log10(hb / ha)) < 0.5, f"DSP FREZ holds: {20*math.log10(hb/ha):+.2f} dB over 1.7 s")
        check(af < hb * 0.1, f"DSP FREZ off decays: {20*math.log10(max(af,1e-12)/hb):+.1f} dB")

        # MIX 0
        x = ref.hits(0.5)
        L, R, _, _ = dsp_render(dict(MIX=0), x, None, tmp)
        x = list(struct.unpack(f"<{len(x)}f", struct.pack(f"<{len(x)}f", *x)))   # what the engine was fed
        def q(v):   # the harness's lround: halves away from zero
            w = max(-1.0, min(1.0 - 2 ** -23, v)) * 2 ** 23
            return int(math.copysign(math.floor(abs(w) + 0.5), w))
        check(all(q(a) == q(b) for a, b in zip(L, x)) and all(q(a) == q(b) for a, b in zip(R, x)),
              "DSP MIX 0 is bit-exact dry")

        # silence after the input stops
        L, R, _, _ = dsp_render(dict(DCAY=60, MIX=127), ref.burst(6.0), None, tmp)
        tail = max(max(abs(v) for v in L[-int(0.5 * FS):]), max(abs(v) for v in R[-int(0.5 * FS):]))
        check(tail < 10 ** (-90 / 20), f"DSP settles: {20*math.log10(max(tail,1e-12)):.0f} dBFS in the last 0.5 s")

    print(f"cost: at most {worst} instructions per 16-frame call = {worst/16:.0f} per frame "
          f"(stock SPRING REV target ~314 per frame; emulator instruction count, not chip cycles)")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
