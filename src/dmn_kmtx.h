/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Cross-process keyed mutex (IDXGIKeyedMutex) over a page of the shared
 * texture allocation. The state page lives at dmn_share_page_align(pod.size)
 * inside the texture's fd — the producer allocates it by arming the texture
 * create with an extra page whenever D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX is
 * set, and a consumer finds it deterministically from the POD (no extra POD
 * field, no version bump).
 *
 * Semantics follow Windows: the mutex starts released with key 0;
 * AcquireSync(K) blocks until it is released with key K and acquires it;
 * ReleaseSync(K) releases it with key K for the next acquirer. Waiting is
 * poll-based with the same backoff as the shared fence (no cross-process
 * futex on macOS).
 *
 * MS-ABI TU (vends a real IDXGIKeyedMutex COM object); dmn_com_hooks.cpp
 * routes texture QueryInterface here.
 */

#pragma once

#include <dxgi1_2.h>

#include <cstdint>

/* Producer side: initialize the state page and register the vended keyed
 * mutex for `texture_identity` (dmn_com_identity of the resource). Consumer
 * side (init=false): attach to an existing page. `fd` is the texture's shm
 * fd; `offset` the page's byte offset. Returns false on failure (the texture
 * still shares; only keyed-mutex sync is unavailable). */
bool dmn_kmtx_register(void* texture_identity, int fd, uint64_t offset, bool init);

/* The vended IDXGIKeyedMutex for a texture identity, AddRef'd, or NULL.
 * Called from the texture QueryInterface hook. */
IDXGIKeyedMutex* dmn_kmtx_lookup(void* texture_identity);

/* Drop the registry's reference when the texture is destroyed. The mutex (and
 * its mapping) lives on until any refs the app still holds are released.
 * No-op for identities without a registered mutex. */
void dmn_kmtx_unregister(void* texture_identity);
