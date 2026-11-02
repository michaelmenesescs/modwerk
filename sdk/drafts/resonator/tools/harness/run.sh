#!/bin/sh
# Build the standalone harness and render one case through the assembled engine.
# Runs INSIDE the modwerk-source-tools image with the module folder mounted at /w:
#   docker run --rm -v "$PWD":/w modwerk-source-tools sh /w/tools/harness/run.sh IN.f32 OUT.f32 EVENTS
set -e
V=/opt/toolchain/vendor/dsp56300
B=/w/tools/harness/.build
mkdir -p "$B"
if [ ! -x "$B/harness" ] || [ /w/tools/harness/harness.cpp -nt "$B/harness" ]; then
  g++ -std=c++17 -O2 -I$V/source -I$V/source/asmjit/src /w/tools/harness/harness.cpp \
    $V/build/source/dsp56kEmu/libdsp56kEmu.a $V/build/source/dsp56kBase/libdsp56kBase.a \
    $V/build/source/asmjit/libasmjit.a $V/build/source/vtuneSdk/libvtuneSdk.a -lpthread -ldl -lrt -o "$B/harness"
fi
sed 's/\$fab1e0/$001000/g' /w/resonator.asm > "$B/r.asm"
$V/build/source/dsp_host/dsp_asm -in "$B/r.asm" -org 2000 -out "$B/r.bin" -sym "$B/r.sym" > "$B/asm.log"
INIT=$(awk '$1=="init"{print $2}' "$B/r.sym")
PROC=$(awk '$1=="proc"{print $2}' "$B/r.sym")
"$B/harness" "$B/r.bin" 2000 "$INIT" "$PROC" /w/tools/harness/.build/ptable.txt 1000 "$1" "$2" "$3"
