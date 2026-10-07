/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * dmn_dxil_inline_subroutines reads a container the guest built; a part
 * offset or size whose 32-bit sum with the header wraps must be rejected,
 * not followed out of the buffer.  Each container below is one DXIL part
 * in the minimal shape (zero digest) with one field pushed past the end,
 * placed against an inaccessible page so a read past it faults.
 *
 * Prints "DXILBOUNDS: PASS" and exits 0 on success.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <sys/mman.h>
#include <unistd.h>

#define T_TAG "DXILBOUNDS"
#include "common/check.h"

extern "C" int dmn_dxil_inline_subroutines(const void* container, size_t size,
                                           void** out, size_t* out_size);

namespace {

/* Header (36) + part header (8) + DxilProgramHeader (24) + 4 bytes. */
const size_t kSize = 72;

/* A program header claiming bitcode far past the container, which only the
 * part size bounds. */
void build(unsigned char* c, uint32_t partOff, uint32_t partSize) {
    memset(c, 0, kSize);
    memcpy(c, "DXBC", 4);
    const uint32_t one = 1, total = kSize;
    memcpy(c + 20, &one, 4);
    memcpy(c + 24, &total, 4);
    memcpy(c + 28, &one, 4);
    memcpy(c + 32, &partOff, 4);
    memcpy(c + 36, "DXIL", 4);
    memcpy(c + 40, &partSize, 4);
    const uint32_t prog[6] = {0x60, 0, 0x4C495844u, 0x100, 16, 0x01000000u};
    memcpy(c + 44, prog, sizeof(prog));
    memcpy(c + 68, "BC\xC0\xDE", 4);
}

} // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const size_t page = (size_t)getpagesize();
    unsigned char* pages = (unsigned char*)mmap(nullptr, 2 * page,
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    EXPECT(pages != MAP_FAILED, "mmap failed");
    EXPECT(mprotect(pages + page, page, PROT_NONE) == 0, "mprotect failed");
    unsigned char* c = pages + page - kSize;
    struct { const char* name; uint32_t partOff, partSize; } cases[] = {
        {"part offset wraps the header check", 0xFFFFFFF0u, 28},
        {"part size wraps the part check", 36, 0xFFFFFFF0u},
    };
    for (const auto& k : cases) {
        build(c, k.partOff, k.partSize);
        void* out = nullptr;
        size_t n = 0;
        const int r = dmn_dxil_inline_subroutines(c, kSize, &out, &n);
        printf(T_TAG ": %s -> %d\n", k.name, r);
        EXPECT(r == 0 && !out, "malformed container was not rejected");
    }
    T_PASS();
    return 0;
}
