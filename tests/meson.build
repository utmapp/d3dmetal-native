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
test('shared-texture', shared_texture_test, env: ['DMN_LOG=info'], timeout: 300)

# Cross-API, cross-process shared fence: forked producer/consumer, one direction
# per test entry (D3D11 fence -> D3D12 consumer, and D3D12 fence -> D3D11 consumer).
fence_xapi = executable('fence-xapi-test',
  'fence_xapi_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('fence-xapi-d11d12', fence_xapi, args: ['d11d12'],
     env: ['DMN_LOG=info'], timeout: 300)
test('fence-xapi-d12d11', fence_xapi, args: ['d12d11'],
     env: ['DMN_LOG=info'], timeout: 300)

# GPU-side wait on an imported fence: a native wait on the real local fence,
# released by the wait watcher only when the producer's GPU writes the value
# into the shared slot.
fence_gpuwait = executable('fence-gpuwait-test',
  'fence_gpuwait_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('fence-gpuwait', fence_gpuwait, env: ['DMN_LOG=info'], timeout: 300)

# Cross-process keyed mutex over a shared texture: forked producer/consumer
# hand the lock back and forth by key to serialize a shared counter.
keyed_mutex = executable('keyed-mutex-test',
  'keyed_mutex_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('keyed-mutex', keyed_mutex, env: ['DMN_LOG=info'], timeout: 300)

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
test('lifecycle-churn', lifecycle_churn, env: ['DMN_LOG=warn'], timeout: 600)

# Multi-threaded stress: concurrent device creation (races the vtable
# patching) and same-device shared create/export/open churn from N threads
# with thread-unique dimensions (detects thread-local arm cross-capture).
mt_stress = executable('mt-stress-test',
  'mt_stress_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('mt-stress', mt_stress, env: ['DMN_LOG=warn'], timeout: 600)

# Windowless: event-driven fence paths — SetEventOnCompletion + dmn_event_wait
# on producer/imported fences, CPU-signal visibility both directions, and
# SetEventOnMultipleFenceCompletion (ALL/ANY) with an imported fence.
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

# Windowless: D3D11 variant shared entry points (CreateTexture2D1 producer,
# OpenSharedResource1 consumer).
d3d11_variants = executable('d3d11-variants-test',
  'd3d11_variants_test.cpp',
  include_directories: test_native_inc,
  link_with: dmn_lib)
test('d3d11-variants', d3d11_variants, env: ['DMN_LOG=info'], timeout: 300)

# Callback-backed swapchain: embedder MTLTextures, configure/acquire/present
# round trip, final clear color verified in the embedder texture.
callback_swapchain = executable('callback-swapchain-test',
  ['callback_swapchain_test.cpp', 'callback_embedder.mm'],
  include_directories: test_native_inc,
  link_with: dmn_lib,
  dependencies: [cocoa_deps])
test('callback-swapchain', callback_swapchain, env: ['DMN_LOG=info'], timeout: 300)
