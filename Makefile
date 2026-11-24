CC ?= gcc
CFLAGS ?= -O3 -std=c99 -Wall -Wextra
LDFLAGS ?= -lm

.PHONY: build test clean android

build:
	$(CC) $(CFLAGS) c_runtime/plm.c $(LDFLAGS) -o c_runtime/plm_run

test:
	python3 -m unittest discover -s tests -v

android:
	@echo "Hasznalat Android NDK-val:"
	@echo "  aarch64-linux-android29-clang -O3 -std=c99 -march=armv8-a c_runtime/plm.c -lm -o c_runtime/plm_android_arm64"

clean:
	rm -f c_runtime/plm_run c_runtime/plm_android_arm64
