/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * DmnGFXTAllocation — memory services: D3DMetal uses these for page
 * allocations and JIT code. Under Rosetta (x86_64) a plain RW -> RX
 * mprotect two-step is what translated JITs expect; no MAP_JIT
 * machinery is involved.
 */

#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>

#include "dmn_gfxt.h"
#include "dmn_log.h"

namespace {

size_t page_align(size_t bytes) {
    size_t page = (size_t)getpagesize();
    return (bytes + page - 1) & ~(page - 1);
}

} // namespace

DmnGFXTAllocation::~DmnGFXTAllocation() = default;

void* DmnGFXTAllocation::allocateBytesFromNewPages(size_t bytes) {
    size_t sz = page_align(bytes);
    void* p = mmap(nullptr, sz, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) {
        DMN_ERROR("allocateBytesFromNewPages(%zu) failed: %s", bytes,
                  strerror(errno));
        return nullptr;
    }
    DMN_TRACE("allocateBytesFromNewPages(%zu) = %p", bytes, p);
    return p;
}

void DmnGFXTAllocation::freeBytesFromAllocatedPages(void* p, size_t bytes) {
    if (!p)
        return;
    DMN_TRACE("freeBytesFromAllocatedPages(%p, %zu)", p, bytes);
    munmap(p, page_align(bytes));
}

void* DmnGFXTAllocation::malloc(size_t bytes) {
    return ::malloc(bytes);
}

void DmnGFXTAllocation::free(void* p) {
    ::free(p);
}

void DmnGFXTAllocation::makeExecutable(void* p, size_t bytes) {
    if (!p || !bytes)
        return;
    uintptr_t page = (uintptr_t)getpagesize();
    uintptr_t base = (uintptr_t)p & ~(page - 1);
    size_t len = page_align(((uintptr_t)p - base) + bytes);
    DMN_TRACE("makeExecutable(%p, %zu)", p, bytes);
    if (mprotect((void*)base, len, PROT_READ | PROT_EXEC) != 0)
        DMN_ERROR("makeExecutable(%p, %zu) mprotect failed: %s", p, bytes,
                  strerror(errno));
}

void* DmnGFXTAllocation::allocateBytesFromImage(const char* moduleName,
                                                size_t bytes) {
    /* The proximity-to-module semantics (if any) are unknown; fresh pages
     * satisfy D3DMetal. Logged at INFO so a real dependency would surface. */
    DMN_INFO("allocateBytesFromImage(%s, %zu) -> new pages",
             moduleName ? moduleName : "(null)", bytes);
    return allocateBytesFromNewPages(bytes);
}
