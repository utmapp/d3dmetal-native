# Several tests run a second process (a forked/exec'd peer, or the churn
# control child) and drive the GPU hard from both. Under meson's default
# parallelism that multiplies the number of concurrent D3D devices until they
# miss their timeouts, so those are marked is_parallel: false.

test_native_inc = include_directories(
  '../include',
  '../include/native/windows',
  '../include/native/directx',
)

cocoa_deps = dependency('appleframeworks',
  modules: ['AppKit', 'QuartzCore', 'Metal', 'MetalKit'])

# Compile-time-determined properties checked as tests (moved out of the
# runtime init path): member-pointer slot decoding, and D3DMetal's
# private-data release-on-destroy contract that registry eviction rides on.
abi_check = executable('abi-check-test', 'abi_check_test.cpp',
  include_directories: [test_native_inc, include_directories('../src')])
test('abi-check', abi_check, timeout: 30)

# The COM invariants every registry in the library keys on: identity resolved
# through a DXGI view must match the resource's and cost no references, and a
# shared resource must die when the app drops it. Needs ../src for the library's
# own dmn_com_identity().
com_contract = executable('com-contract-test', 'com_contract_test.cpp',
  include_directories: [test_native_inc, include_directories('../src')],
  link_with: dmn_lib)
test('com-contract', com_contract, env: ['DMN_LOG=info'],
     is_parallel: false, timeout: 300)

evict_contract = executable('evict-contract-test', 'evict_contract_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('evict-contract', evict_contract, env: ['DMN_LOG=info'], timeout: 300)

dmn_smoke = executable('dmn-smoke', 'dmn_smoke.c',
  include_directories: include_directories('../include'),
  link_with: dmn_lib)
test('smoke', dmn_smoke, env: ['DMN_LOG=info'], timeout: 120)

dmn_device_test = executable('dmn-device-test',
  ['dmn_device_test.cpp', 'cocoa_window.mm'],
  include_directories: test_native_inc,
  link_with: dmn_lib,
  dependencies: [cocoa_deps])
test('device', dmn_device_test, env: ['DMN_LOG=info'], timeout: 300)

format_caps_test = executable('format-caps-test',
  'format_caps_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('format-caps', format_caps_test, env: ['DMN_LOG=info'], timeout: 300)

# What container shapes does D3DMetal accept for SM 6.0 DXIL shaders?  The
# guest UMD receives BARE DXIL program parts from the Windows runtime and must
# re-wrap them; this measures how much container that wrapper needs.
dxil_container_probe = executable('dxil-container-probe',
  'dxil_container_probe.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('dxil-container-probe', dxil_container_probe, env: ['DMN_LOG=info'],
     timeout: 300)

# Windowless: CreateBlendState1 must hand back a usable blend state. Also
# reports what the framework does unaided, so an upstream fix — after which the
# dmn_com_hooks repair is dead code — shows up in the test log.
blend_state1_test = executable('blend-state1-test',
  'blend_state1_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('blend-state1', blend_state1_test, env: ['DMN_LOG=info'], timeout: 300)

d3d11_triangle = executable('d3d11-triangle',
  ['d3d11_triangle.cpp', 'cocoa_window.mm'],
  include_directories: test_native_inc,
  link_with: dmn_lib,
  dependencies: [cocoa_deps])

d3d11_cube = executable('d3d11-cube',
  ['d3d11_cube.cpp', 'cocoa_window.mm'],
  include_directories: test_native_inc,
  link_with: dmn_lib,
  dependencies: [cocoa_deps])

d3d12_triangle = executable('d3d12-triangle',
  ['d3d12_triangle.cpp', 'cocoa_window.mm'],
  include_directories: test_native_inc,
  link_with: dmn_lib,
  dependencies: [cocoa_deps])

# Windowed end-to-end demo: D3D12 compute fills a shared texture, a D3D11
# triangle samples it, fence-gated. Forks a D3D12 producer; parent is the
# windowed D3D11 consumer. Has a headless readback self-check (VERIFY: PASS).
shared_compute_triangle = executable('shared-compute-triangle',
  ['shared_compute_triangle.cpp', 'cocoa_window.mm'],
  include_directories: test_native_inc,
  link_with: dmn_lib,
  dependencies: [cocoa_deps])
# Headless self-check of the data path (no window/present); run without
# DMN_HEADLESS for the on-screen demo.
test('shared-compute-triangle', shared_compute_triangle,
     is_parallel: false,
     env: ['DMN_LOG=warn', 'DMN_HEADLESS=1', 'DMN_INTERVAL_MS=80', 'DMN_FRAMES=400'],
     timeout: 300)

# Windowless: shared D3D12 buffer via standard APIs; GPU WriteBufferImmediate,
# read back through the handle's fd (the GPU-written-fence building block).
d3d12_sharedbuf = executable('d3d12-sharedbuf-test',
  'd3d12_sharedbuf_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('d3d12-sharedbuf', d3d12_sharedbuf, env: ['DMN_LOG=info'], timeout: 300)

d3d11_sharedbuf = executable('d3d11-sharedbuf-test',
  'd3d11_sharedbuf_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('d3d11-sharedbuf', d3d11_sharedbuf, env: ['DMN_LOG=info'], timeout: 300)

# Windowless: D3D12 producer fence (queue Signal -> helper-queue Wait +
# WriteBufferImmediate) read back by the raw consumer and an imported ID3D12Fence.
d3d12_fence = executable('d3d12-fence-test',
  'd3d12_fence_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('d3d12-fence', d3d12_fence, env: ['DMN_LOG=info'], timeout: 300)

# Windowless producer (D3D11) + pure consumer, forked in one binary; no Cocoa.
shared_texture_test = executable('shared-texture-test',
  'shared_texture_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('shared-texture', shared_texture_test, env: ['DMN_LOG=info'],
     is_parallel: false, timeout: 300)

# Windowless, no pixels: MISC_SHARED create/open contract — unshareable
# formats fail the create instead of silently unsharing, imports validate the
# wire geometry they are handed, and a rejected import reports its own failure
# rather than falling through to D3DMetal as "not one of our handles".
shared_contract_test = executable('shared-contract-test',
  'shared_contract_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('shared-contract', shared_contract_test, env: ['DMN_LOG=info'], timeout: 300)

# Regression: rendering to an opened shared surface must not discard the
# contents it already had (create-with-no-initial-data marks it undefined, so
# the first attach loads with a clear).
shared_open_preserve = executable('shared-open-preserve-test',
  'shared_open_preserve_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('shared-open-preserve', shared_open_preserve,
     env: ['DMN_LOG=info'], timeout: 300)

# Regression: shader-sampling a MISC_SHARED texture's SRV (glyph-box bug).
shared_srv_sample_test = executable('shared-srv-sample-test',
  'shared_srv_sample_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('shared-srv-sample', shared_srv_sample_test, env: ['DMN_LOG=info'], timeout: 300)
# The same sample with the impostor residency set switched off, so the
# per-command-buffer useResources: path (macOS < 15, sparse textures) keeps a
# bindlessly bound impostor resident too.
test('shared-srv-sample-no-residency-set', shared_srv_sample_test,
     env: ['DMN_LOG=info', 'DMN_RESIDENCY_SET=0'], timeout: 300)

# A rejected call must leave nothing behind: no crash, no fds, no threads.
# Interception starts work before it knows the call will succeed, so this is
# where an un-unwound arm or watcher shows up.
error_paths = executable('error-paths-test',
  'error_paths_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('error-paths', error_paths, env: ['DMN_LOG=warn'],
     is_parallel: false, timeout: 300)

# Cross-API, cross-process shared fence: forked producer/consumer, one direction
# per test entry (D3D11 fence -> D3D12 consumer, and D3D12 fence -> D3D11 consumer).
fence_xapi = executable('fence-xapi-test',
  'fence_xapi_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('fence-xapi-d11d12', fence_xapi, args: ['d11d12'],
     env: ['DMN_LOG=info'], is_parallel: false, timeout: 300)
test('fence-xapi-d12d11', fence_xapi, args: ['d12d11'],
     env: ['DMN_LOG=info'], is_parallel: false, timeout: 300)

# Regression: a fence must outlive its own signal. Signal-then-drop without
# flushing, repeatedly, then confirm the queue still retires work — destroying a
# signalled fence early kills the whole Metal queue on some versions, which shows
# up as an unkillable process rather than a failed assertion.
fence_release_race = executable('fence-release-race-test',
  'fence_release_race_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('fence-release-race', fence_release_race, env: ['DMN_LOG=warn'],
     is_parallel: false, timeout: 600)

# GPU-side wait on an imported fence: a native wait on the real local fence,
# released by the wait watcher only when the producer's GPU writes the value
# into the shared slot.
fence_gpuwait = executable('fence-gpuwait-test',
  'fence_gpuwait_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('fence-gpuwait', fence_gpuwait, env: ['DMN_LOG=info'],
     is_parallel: false, timeout: 300)

# Cross-process keyed mutex over a shared texture: forked producer/consumer
# hand the lock back and forth by key to serialize a shared counter.
keyed_mutex = executable('keyed-mutex-test',
  'keyed_mutex_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('keyed-mutex', keyed_mutex, env: ['DMN_LOG=info'],
     is_parallel: false, timeout: 300)

# Windowless: create -> open -> release-creator -> re-export (both APIs);
# the opened resource's registration must own its own fd (MSDN handle rule).
shared_reexport = executable('shared-reexport-test',
  'shared_reexport_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('shared-reexport', shared_reexport, env: ['DMN_LOG=info'], timeout: 300)

# Windowless: shared texture/buffer/fence create-destroy churn; asserts fd
# count and memory footprint stay flat (destruction-driven eviction).
lifecycle_churn = executable('lifecycle-churn-test',
  'lifecycle_churn_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('lifecycle-churn', lifecycle_churn, env: ['DMN_LOG=warn'],
     is_parallel: false, timeout: 600)

# Multi-threaded stress: concurrent device creation (races the vtable
# patching) and same-device shared create/export/open churn from N threads
# with thread-unique dimensions (detects thread-local arm cross-capture).
mt_stress = executable('mt-stress-test',
  'mt_stress_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('mt-stress', mt_stress, env: ['DMN_LOG=warn'],
     is_parallel: false, timeout: 600)

# Windowless, no init: public event API semantics — manual/auto reset,
# initial state, wake-one auto-reset, and the pollable dup_fd view.
event_api = executable('event-api-test', 'event_api_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('event-api', event_api, env: ['DMN_LOG=info'], timeout: 120)

# Windowless: event-driven fence paths — SetEventOnCompletion + dmn_event_wait
# on producer/imported fences, CPU-signal visibility both directions,
# SetEventOnMultipleFenceCompletion (ALL/ANY) with an imported fence, and a
# dup_fd polled readable on fence completion.
fence_events = executable('fence-events-test',
  'fence_events_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('fence-events', fence_events, env: ['DMN_LOG=info'], timeout: 300)

# Windowless: D3D12 shared heap + placed resource (the heap-class swizzle
# path), plus CreateCommittedResource1 when ID3D12Device4 is exposed.
heap_placed = executable('heap-placed-test',
  'heap_placed_test.cpp',
  include_directories: [test_native_inc, include_directories('../src')],
  link_with: dmn_lib)
test('heap-placed', heap_placed, env: ['DMN_LOG=info'], timeout: 300)

# Windowless: dmn_open_existing_heap_from_fd — a shm window imported as an
# ID3D12Heap whose placed buffers alias the caller's pages in both directions
# (import offset + heap offset compose), the backing survives releasing the
# heap first, fds return to baseline, and the validation paths refuse cleanly.
heap_import = executable('heap-import-test',
  'heap_import_test.cpp',
  include_directories: [test_native_inc, include_directories('../src')],
  link_with: dmn_lib)
test('heap-import', heap_import, env: ['DMN_LOG=info'], timeout: 300)

# Windowless: how much memory does each shared path actually cost? Holds N large
# resources down every hooked producer/export/import entry point, with an
# unshared control beside each, and measures both the bytes RESERVED and the
# bytes the machine is actually HOLDING. Catches paths where the substitution is
# additive rather than substitutive (one resource, two allocations) and paths
# where placed resources fail to alias — neither of which any other test can
# see, since they all check the bytes and never how many allocations hold them.
#
# Slow by construction: each case is measured twice in-process for the reserved
# figure, and once more in a FRESH CHILD process for the resident figure,
# because freed pages are not reclaimed until something needs them and only
# process exit returns them unconditionally. Hence the long timeout.
#
# FAILS on resident memory (DMN_FP_MAX_RESIDENT, default 1.6x) and on a
# non-aliasing import; over-reservation that stays virtual is reported, not
# fatal. DMN_FP_REPORT=1 downgrades to report-only, DMN_FP_STRICT=1 also fails
# on warnings.
shared_footprint = executable('shared-footprint-test',
  ['shared_footprint_test.cpp', 'metal_alloc_size.mm'],
  include_directories: [test_native_inc, include_directories('../src')],
  link_with: dmn_lib,
  dependencies: [cocoa_deps])
test('shared-footprint', shared_footprint, env: ['DMN_LOG=error'],
     is_parallel: false, timeout: 2400)

# Windowless: D3D11 variant shared entry points (CreateTexture2D1 producer,
# OpenSharedResource1 consumer).
d3d11_variants = executable('d3d11-variants-test',
  'd3d11_variants_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('d3d11-variants', d3d11_variants, env: ['DMN_LOG=info'], timeout: 300)

# Diagnostic probe (not a test): D3DMetal pool behaviour on release.
pool_probe = executable('pool-probe',
  ['pool_probe.cpp', 'metal_alloc_size.mm'],
  include_directories: [test_native_inc, include_directories('../src')],
  link_with: dmn_lib,
  dependencies: [cocoa_deps])

# Diagnostic probe (not a test): the backend's own tiled-resource support.
tiled_probe = executable('tiled-probe',
  ['tiled_probe.cpp', 'metal_alloc_size.mm'],
  include_directories: [test_native_inc, include_directories('../src')],
  link_with: dmn_lib,
  dependencies: [cocoa_deps])

# Diagnostic probe (not a test): what the machine actually holds for committed
# textures vs how much of them the GPU touches (host-wide page counters), and
# the A/B of a reserved texture under committed vs sparse backing.
resident_probe = executable('resident-probe',
  ['resident_probe.cpp', 'metal_alloc_size.mm'],
  include_directories: [test_native_inc, include_directories('../src')],
  link_with: dmn_lib,
  dependencies: [cocoa_deps])
# Diagnostic probe (not a test): is an unreferenced D3D12 heap / untouched
# texture still made resident by the backend on GPU work? (Yes.)
heapres_probe = executable('heapres-probe',
  ['heapres_probe.cpp', 'metal_alloc_size.mm'],
  include_directories: [test_native_inc, include_directories('../src')],
  link_with: dmn_lib,
  dependencies: [cocoa_deps])

# Sparse-backed reserved resources: tier-2 content semantics through the
# hooks (mapped region holds data, unmapped reads zero, unmap reads zero
# again; BC7 4096^2 chain incl. every packed-tail mip written and read back).
sparse_tiled = executable('sparse-tiled-test',
  ['sparse_tiled_test.cpp', 'metal_alloc_size.mm'],
  include_directories: [test_native_inc, include_directories('../src')],
  link_with: dmn_lib,
  dependencies: [cocoa_deps])
test('sparse-tiled', sparse_tiled, env: ['DMN_LOG=warn'], is_parallel: false, timeout: 120)

# How the sparse module sizes itself: when chunks open, where a texture lands,
# what the chunks may total.  Each case re-execs this binary in a fresh process
# (the knobs are read once and the chunks are process-global), so it must not
# run alongside other GPU tests.
sparse_sizing = executable('sparse-sizing-test',
  ['sparse_sizing_test.cpp', 'metal_alloc_size.mm'],
  include_directories: [test_native_inc, include_directories('../src')],
  link_with: dmn_lib,
  dependencies: [cocoa_deps])
test('sparse-sizing', sparse_sizing, env: ['DMN_LOG=warn'], is_parallel: false, timeout: 300)

# Diagnostic probe (not a test): a streaming title's reserved-resource pattern
# -- create many tiled resources, then map a fraction of each much later --
# and whether any sparse chunk runs dry.  Reproduces the shape that stresses
# placement on the host in under a minute, with no VM.
sparse_pool_probe = executable('sparse-pool-probe',
  ['sparse_pool_probe.cpp', 'metal_alloc_size.mm'],
  include_directories: [test_native_inc, include_directories('../src')],
  link_with: dmn_lib,
  dependencies: [cocoa_deps])

