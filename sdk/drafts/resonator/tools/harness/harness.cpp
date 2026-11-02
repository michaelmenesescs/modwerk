// SPDX-License-Identifier: MIT
// Standalone DSP56300 harness for RESONATOR: the engine alone, no firmware.
//
// Loads the assembled engine at P:CODE and its ptable at P:TBL, puts the
// instance where the dispatcher would (r7 state block, allocator entry at
// x:$213, a 16K Y buffer at $4000 filled with garbage first), runs init, then
// calls proc once per 16-frame block with r0/n7/r6/r7 set as the dispatcher
// sets them. Knob events are applied to the parameter block between calls,
// with page 2's slots packed knob|companion as on the unit.
//
//   harness BLOB CODE INIT PROC PTABLE TBL IN.f32 OUT.f32 EVENTS
//     IN.f32  mono float32 input; OUT.f32 stereo interleaved float32
//     EVENTS  lines "frame slot value" (slot 0..11, value 0..127)
// Prints the most instructions any proc call took and whether any Y word
// outside the rings changed.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "dsp56kEmu/dsp.h"
#include "dsp56kEmu/memory.h"
#include "dsp56kEmu/peripherals.h"

using namespace dsp56k;

class AllowAll : public IMemoryValidator {
public:
    bool memValidateAccess(EMemArea, TWord, bool) const override { return true; }
};

static const TWord SENTINEL = 0x03f000;
static const TWord PBLOCK = 0x7000, STATE = 0x6200, AUDIO = 0x7100, ALLOC = 0x0300;
static const TWord YBASE = 0x4000, YEND = 0x8000, RINGS_END = 0x6000;

static long runToRts(DSP& dsp, TWord pc, long maxSteps = 20000000) {
    dsp.setPC(SENTINEL);
    dsp.jsr(pc);
    const uint64_t i0 = dsp.getInstructionCounter();
    for (long i = 0; i < maxSteps; ++i) {
        if (dsp.getPC().toWord() == SENTINEL) return static_cast<long>(dsp.getInstructionCounter() - i0);
        dsp.execInterpreter();
    }
    std::fprintf(stderr, "no return from %06x (pc=%06x)\n", pc, dsp.getPC().toWord());
    return -1;
}

static TWord q23(float v) {
    double d = std::max(-1.0, std::min(1.0 - 1.0 / 8388608.0, (double)v));
    long w = lround(d * 8388608.0);
    return static_cast<TWord>(w) & 0xffffff;
}
static float fromq23(TWord w) {
    long v = (w & 0x800000) ? (long)w - 0x1000000 : (long)w;
    return static_cast<float>(v / 8388608.0);
}

int main(int argc, char** argv) {
    if (argc < 10) { std::fprintf(stderr, "usage: harness BLOB CODE INIT PROC PTABLE TBL IN OUT EVENTS\n"); return 2; }
    const TWord code = strtoul(argv[2], nullptr, 16), init = strtoul(argv[3], nullptr, 16),
                proc = strtoul(argv[4], nullptr, 16), tbl = strtoul(argv[6], nullptr, 16);
    AllowAll validator;
    Memory mem(validator, 0x080000, 0x800000, 0x200000);
    Peripherals56362 px;
    Peripherals56367 py;
    DSP dsp(mem, &px, &py);

    std::ifstream bf(argv[1], std::ios::binary);
    std::vector<unsigned char> blob((std::istreambuf_iterator<char>(bf)), {});
    for (size_t i = 0; i + 2 < blob.size(); i += 3)
        mem.set(MemArea_P, code + i / 3, blob[i] | (blob[i + 1] << 8) | (blob[i + 2] << 16));
    std::ifstream tf(argv[5]);
    TWord w, at = tbl;
    while (tf >> std::hex >> w) mem.set(MemArea_P, at++, w);

    // garbage everywhere the engine may read before it writes
    unsigned seed = 12345;
    auto rnd = [&]() { seed = seed * 1103515245u + 12345u; return (seed >> 8) & 0xffffff; };
    for (TWord a = YBASE; a < YEND; ++a) mem.set(MemArea_Y, a, rnd());
    for (TWord a = STATE; a < STATE + 0x100; ++a) mem.set(MemArea_X, a, rnd());
    std::vector<TWord> ysnap;
    for (TWord a = RINGS_END; a < YEND; ++a) ysnap.push_back(mem.get(MemArea_Y, a));
    mem.set(MemArea_X, 0x213, ALLOC);
    mem.set(MemArea_X, ALLOC, YBASE);

    std::ifstream inf(argv[7], std::ios::binary);
    std::vector<float> in;
    float s;
    while (inf.read(reinterpret_cast<char*>(&s), 4)) in.push_back(s);

    struct Ev { long frame; int slot; int value; };
    std::vector<Ev> ev;
    std::ifstream ef(argv[9]);
    long fr; int sl, va;
    while (ef >> std::dec >> fr >> sl >> va) ev.push_back({fr, sl, va});
    std::stable_sort(ev.begin(), ev.end(), [](const Ev& a, const Ev& b) { return a.frame < b.frame; });
    int knob[12] = {0};
    size_t ei = 0;
    auto apply = [&](long upto) {
        while (ei < ev.size() && ev[ei].frame <= upto) { knob[ev[ei].slot] = ev[ei].value & 0x7f; ++ei; }
        for (int i = 0; i < 6; ++i) mem.set(MemArea_X, PBLOCK + i, (TWord)knob[i] << 16);
        for (int k = 0; k < 3; ++k)
            mem.set(MemArea_X, PBLOCK + 0xc + k, ((TWord)knob[6 + 2 * k] << 16) | ((TWord)knob[7 + 2 * k] << 8));
    };

    dsp.regs().r[7].var = STATE;
    dsp.regs().r[6].var = PBLOCK;
    if (runToRts(dsp, init) < 0) return 1;

    std::vector<float> out;
    long maxInstr = 0, sumInstr = 0, calls = 0;
    for (size_t f = 0; f < in.size(); f += 16) {
        apply(static_cast<long>(f));
        const size_t n = std::min<size_t>(16, in.size() - f);
        for (size_t i = 0; i < n; ++i) {
            mem.set(MemArea_X, AUDIO + 2 * i, q23(in[f + i]));
            mem.set(MemArea_X, AUDIO + 2 * i + 1, q23(in[f + i]));
        }
        dsp.regs().r[0].var = AUDIO;
        dsp.regs().r[6].var = PBLOCK;
        dsp.regs().r[7].var = STATE;
        dsp.regs().n[7].var = static_cast<TWord>(n);
        long st = runToRts(dsp, proc);
        if (st < 0) return 1;
        if (calls >= 64) { maxInstr = std::max(maxInstr, st); sumInstr += st; }
        ++calls;
        for (size_t i = 0; i < n; ++i) {
            out.push_back(fromq23(mem.get(MemArea_X, AUDIO + 2 * i)));
            out.push_back(fromq23(mem.get(MemArea_X, AUDIO + 2 * i + 1)));
        }
    }
    int stray = 0;
    for (TWord a = RINGS_END; a < YEND; ++a) if (mem.get(MemArea_Y, a) != ysnap[a - RINGS_END]) ++stray;
    std::ofstream of(argv[8], std::ios::binary);
    of.write(reinterpret_cast<const char*>(out.data()), out.size() * 4);
    std::printf("calls %ld  live-call instructions max %ld  mean %.1f  (per frame max %.1f)  stray Y writes %d\n",
                calls, maxInstr, calls > 64 ? (double)sumInstr / (calls - 64) : 0.0, maxInstr / 16.0, stray);
    return stray ? 3 : 0;
}
