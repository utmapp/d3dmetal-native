/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * dmn_dxil_normalize reads a container the guest built; a part offset or size
 * whose 32-bit sum with the header wraps must be rejected, not followed out of
 * the buffer. Each container below is one DXIL part in the minimal shape
 * (zero digest) with one field pushed past the end, placed against an
 * inaccessible page so a read past it faults.
 *
 * Prints "DXILBOUNDS: PASS" and exits 0 on success.
 */

#include <cerrno>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#define T_TAG "DXILBOUNDS"
#include "common/check.h"

extern "C" int dmn_dxil_normalize(const void* container, size_t size,
                                  void** out, size_t* out_size);
typedef unsigned char BYTE;
#include "shaders/quadvote0.h"

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
/* A duplicate table must not multiply decoding work by the part count.
 * Bound CPU time in a child rather than using a wall-clock timing assertion. */
bool duplicatePartWorkBound() {
    uint32_t originalCount;
    memcpy(&originalCount, g_quadvote0 + 28, 4);
    uint32_t originalOffset = 0, chunkSize = 0;
    for (uint32_t i = 0; i < originalCount; ++i) {
        uint32_t offset;
        memcpy(&offset, g_quadvote0 + 32 + i * 4, 4);
        if (memcmp(g_quadvote0 + offset, "DXIL", 4) == 0) {
            originalOffset = offset;
            memcpy(&chunkSize, g_quadvote0 + offset + 4, 4);
            chunkSize += 8;
            break;
        }
    }
    if (!chunkSize)
        return false;
    const uint32_t count = 1u << 20, offset = 32 + count * 4;
    const uint32_t total = offset + chunkSize, version = 1;
    std::vector<unsigned char> container(total);
    memcpy(container.data(), "DXBC", 4);
    memcpy(container.data() + 20, &version, 4);
    memcpy(container.data() + 24, &total, 4);
    memcpy(container.data() + 28, &count, 4);
    for (uint32_t i = 0; i < count; ++i)
        memcpy(container.data() + 32 + i * 4, &offset, 4);
    memcpy(container.data() + offset, g_quadvote0 + originalOffset, chunkSize);
    const pid_t child = fork();
    if (child < 0)
        return false;
    if (child == 0) {
        const rlimit cpu = {1, 1};
        if (setrlimit(RLIMIT_CPU, &cpu) != 0)
            _exit(2);
        void *out = nullptr;
        size_t size = 0;
        const int changed = dmn_dxil_normalize(
            container.data(), container.size(), &out, &size);
        const bool rejected = changed == 0 && !out;
        free(out);
        _exit(rejected ? 0 : 1);
    }
    int status = 0;
    pid_t waited;
    do {
        waited = waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited != child)
        return false;
    printf(T_TAG ": duplicate-part work bound: exit=%d signal=%d\n",
           WIFEXITED(status) ? WEXITSTATUS(status) : -1,
           WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
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
        const int r = dmn_dxil_normalize(c, kSize, &out, &n);
        printf(T_TAG ": %s -> %d\n", k.name, r);
        EXPECT(r == 0 && !out, "malformed container was not rejected");
    }
    EXPECT(duplicatePartWorkBound(), "duplicate parts exceeded bounded rejection work");
    T_PASS();
    return 0;
}
