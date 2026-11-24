# PLM futtatása Android telefonon (Termux & Android NDK) 📱

A `c_runtime/plm.c` tiszta, **függőségmentes C99** kód — semmilyen külső könyvtárra nincs szüksége (csak a szabványos `libm`-re), ezért Android ARM64 telefonon közvetlenül lefordítható és futtatható.

## 1. opció — Közvetlenül telefonon (Termux, 1 perc)

Telepítsd a Termux alkalmazást (F-Droidról), majd futtasd:

```bash
pkg update && pkg install -y git clang make python
git clone https://github.com/PekRichie/plm.git
cd plm
make CC=clang build
python3 -m unittest discover -s tests -v
```

## 2. opció — Keresztfordítás Android NDK-val (gépről ARM64 binárisra)

```bash
$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang \
  -O3 -std=c99 -march=armv8-a c_runtime/plm.c -lm -o c_runtime/plm_android_arm64
```

Majd másold a telefonra `adb push` segítségével:

```bash
adb push c_runtime/plm_android_arm64 /data/local/tmp/plm_run
adb shell chmod +x /data/local/tmp/plm_run
```

## Mobil RAM-keret (ajánlott)

| Preset | Paraméterek | Q4_0 súlyméret | Minimális telefon RAM |
|---|---:|---:|---:|
| `nano-30m` | ~26M | ~14 MB | 2 GB |
| `micro-180m` | ~158M | ~85 MB | 4 GB |
