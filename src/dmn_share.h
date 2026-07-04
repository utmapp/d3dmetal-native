/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Internal facade for cross-process resource sharing. Split cleanly so the
 * COM-hook TU (dmn_com_hooks.cpp, plain C++ + DirectX headers, MS-ABI) never
 * touches Metal, and the Metal TU (dmn_share_metal.mm, ObjC++/SysV) never
 * touches the DirectX headers. They communicate only through the POD structs
 * and plain-C entry points declared here.
 *
 * The "arm" mechanism: a producer/consumer COM thunk, just before it calls a
 * D3DMetal create that will internally allocate a Metal texture, arms a
 * thread-local record. The ObjC swizzle installed over the Metal device/heap
 * classes sees the armed record on the very next texture creation, substitutes
 * a shared-memory-backed linear texture, records what it built, and disarms.
 */

#pragma once

#include <cstddef>
#include <cstdint>

/* Internal mirror of the public PODs (include/d3dmetal_native.h). Kept as a
 * separate declaration so this header has no include-order dependency; the
 * static_asserts in dmn_share_metal.mm check the layouts stay identical. */
struct DmnShareTexPOD {
    uint32_t magic, version;
    int32_t  fd;
    uint32_t width, height;
    uint32_t dxgi_format;
    uint32_t mip_levels, array_size, sample_count;
    uint32_t bind_flags, misc_flags, cpu_access;
    uint64_t stride;
    uint64_t size;
};

struct DmnShareFencePOD {
    uint32_t magic, version;
    int32_t  fd;
    uint32_t flags;
    uint64_t initial_value;
};

/* == Arm record (thread-local; set by hooks, consumed by the swizzle) ===== */

enum DmnShareKind {
    DMN_SHARE_TEXTURE = 0,
    DMN_SHARE_BUFFER  = 1,  /* raw MTLBuffer (e.g. a GPU-written fence-value page) */
};

struct DmnShareArm {
    bool     armed;
    int      kind;          /* DmnShareKind; texture and buffer swizzles ignore
                               an arm whose kind isn't theirs */
    bool     alloc_new;     /* true: producer allocates; false: consumer reuses fd */
    uint64_t extra_bytes;   /* producer: extra shm past the page-aligned payload
                               (e.g. the keyed-mutex page), page-aligned itself */
    int      existing_fd;   /* consumer: fd to mmap (byte-exact layout) */
    uint64_t existing_stride;
    uint64_t existing_size;

    /* Filled by the swizzle on capture. */
    bool     captured;
    int      out_fd;
    uint64_t out_stride;
    uint64_t out_size;
};

/* Arm the calling thread for the next Metal texture creation. extra_bytes
 * requests additional shared memory past the page-aligned texture bytes
 * (found by a consumer at page_align(pod.size)); 0 for none. */
void dmn_share_arm_producer(uint64_t extra_bytes);
void dmn_share_arm_consumer(int fd, uint64_t stride, uint64_t size);
/* Arm the calling thread for the next raw MTLBuffer creation (device
 * newBufferWithLength:options: or heap newBufferWithLength:options:offset:).
 * `size` is the byte length to back with shared memory. */
void dmn_share_arm_producer_buffer(uint64_t size);
void dmn_share_arm_consumer_buffer(int fd, uint64_t size);
/* Disarm; copies the capture result into *out (may be NULL). Returns whether a
 * substitution was actually captured. Logs loudly if armed-but-not-captured. */
bool dmn_share_disarm(DmnShareArm* out);
/* Whether the calling thread currently has an arm pending. Lets a re-entrant
 * COM hook (e.g. CreateCommittedResource called from a consumer reconstruct)
 * pass straight through instead of re-detecting a producer create. */
bool dmn_share_is_armed(void);

/* Install the device/heap ObjC swizzles. Idempotent, must run before D3DMetal
 * loads. No-op-safe to call repeatedly. */
void dmn_share_install_swizzles(void);

/* == Shared-memory helpers used by the C-API + fence (no Metal needed) ==== */

/* mmap the fd PROT_READ|PROT_WRITE, MAP_SHARED. Returns NULL on failure. */
void* dmn_share_map_fd(int fd, size_t size);
void  dmn_share_unmap(void* ptr, size_t size);

/* Round up to the page size (the trailer-offset rule shared by producer
 * allocation and consumer lookup). */
size_t dmn_share_page_align(size_t n);
