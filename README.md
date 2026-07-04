# d3dmetal-native

Direct3D 11 and Direct3D 12 on macOS for **native (non-Wine) applications**,
via Apple's `D3DMetal.framework` from the [Game Porting Toolkit].

D3DMetal implements D3D11/D3D12/DXGI on top of Metal, but it is built to be
hosted by Apple's Wine environment: it expects a `GFXT` host interface for
windowing, events, registry, and memory services. This library implements that
host interface natively — link `libd3dmetal-native.dylib`, and the exported
D3D entry points (`D3D11CreateDevice`, `D3D12CreateDevice`,
`CreateDXGIFactory1`, `D3DCompile`, ...) just work in an ordinary macOS
process.

```
your app ──> libd3dmetal-native.dylib ──> D3DMetal.framework ──> Metal
             (GFXT host: windows, events,
              registry, allocation, swapchains,
              cross-process resource sharing)
```

[Game Porting Toolkit]: https://developer.apple.com/games/game-porting-toolkit/

## Features

- **D3D11 and D3D12 devices** with swapchains on:
  - an `NSView` (the library installs a `CAMetalLayer`),
  - a caller-owned `CAMetalLayer` (offscreen or custom layer hierarchies), or
  - an **exported swapchain** (`dmn_window_create_exported`) — for
    compositors, VM display servers, and remoting: the library allocates
    shared-memory-backed images and drives three callbacks
    (images-changed with the fds/strides, acquire for backpressure,
    present with a pollable GPU-done fence fd). The embedder never
    touches a Metal object.
- **Pseudo-HWNDs**: `dmn_window_get_hwnd()` returns a small stable handle to
  pass anywhere a `HWND` is expected (`CreateSwapChainForHwnd`, ...).
- **Cross-process resource sharing**, entirely through the standard D3D APIs
  (D3DMetal itself stubs these out; the library re-implements them):
  - shared **textures** and **buffers** (`D3D11_RESOURCE_MISC_SHARED*`,
    `D3D12_HEAP_FLAG_SHARED`, committed and placed resources), exported with
    `GetSharedHandle` / `CreateSharedHandle` and opened with
    `OpenSharedResource(1)` / `OpenSharedHandle` — cross-API in both
    directions (D3D11 producer → D3D12 consumer and vice versa);
  - shared **fences** (`CreateFence` + `CreateSharedHandle`,
    `OpenSharedFence` / `OpenSharedHandle`) with GPU-side queue/context
    Wait and Signal, `SetEventOnCompletion`, and multi-fence waits;
  - **keyed mutexes** (`D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX` +
    `IDXGIKeyedMutex` via `QueryInterface`).
- **Win32-style events**: HANDLEs surfaced through D3D APIs (frame-latency
  waitable objects, `SetEventOnCompletion`) are waitable with
  `dmn_event_wait()`. `dmn_event_create()` mints HANDLEs to pass into those
  APIs yourself, and `dmn_event_dup_fd()` vends a `poll(2)`-compatible fd
  mirroring an event's signaled state (for fd-driven event loops).
- **Registry pre-seeding** (`dmn_registry_set_*`) and per-value environment
  overrides for the settings D3DMetal reads.
- **Executable-path override** (`dmn_set_executable_path`) so D3DMetal's
  per-app profile matcher keys on the program you are hosting rather than
  your own process.

## Requirements

- macOS 14+.
- **x86_64 only**: `D3DMetal.framework` ships as x86_64, so the entire process
  must be x86_64. On Apple Silicon everything runs under Rosetta 2.
- `D3DMetal.framework` from Apple's Game Porting Toolkit. It is Apple-licensed
  and NOT bundled; obtain it from the GPTk evaluation environment and point
  the library at it (see below). If macOS quarantines it:
  `xattr -dr com.apple.quarantine D3DMetal.framework`.

## Building

```sh
meson setup build --cross-file build-macos-x86_64.txt
meson compile -C build
meson test -C build          # optional; needs a D3DMetal.framework (see below)
```

The cross file is required on Apple Silicon hosts (it selects
`-arch x86_64`; Rosetta runs the results, including the tests, transparently).

## Using it

1. **Link against `libd3dmetal-native.dylib` only.** Do not link or `dlopen`
   D3DMetal yourself; the library locates it, loads it, and installs the GFXT
   host interface. The search order for the framework:
   - `dmn_options::framework_path` passed to `dmn_init()`,
   - `$D3DMETAL_FRAMEWORK_PATH`,
   - next to the dylib (`<dylib dir>/D3DMetal.framework`), then
     `<dylib dir>/../Frameworks/D3DMetal.framework`.
2. **Bring your own D3D headers** (Wine/MinGW widl-generated). The D3D/DXGI
   symbols resolve from this library at link time. TUs that include `d3d12.h`
   must be compiled with `-DWIDL_EXPLICIT_AGGREGATE_RETURNS=1` to match
   D3DMetal's ABI for aggregate-returning methods (`GetDesc`, ...).
3. **Create a window object** for your view or layer and use its pseudo-HWND:

```c
#include <d3dmetal_native.h>

dmn_window_t win = dmn_window_create_for_view(nsview); /* main thread */
dmn_window_set_size(win, pixel_w, pixel_h, scale);
HWND hwnd = (HWND)dmn_window_get_hwnd(win);
/* ... D3D11CreateDevice / CreateDXGIFactory1 /
   IDXGIFactory2::CreateSwapChainForHwnd(dev, hwnd, ...) as on Windows ... */
```

`dmn_init()` is optional — the first D3D entry point performs lazy default
initialization. See `include/d3dmetal_native.h` for the full API, including
the exported swapchain (`dmn_window_create_exported`) and the shared-handle
lifetime rules.

### Cross-process sharing model

A `HANDLE` returned by the `GetSharedHandle` / `CreateSharedHandle` family is
a pointer to a POD (`dmn_shared_texture_handle`, `dmn_shared_buffer_handle`,
`dmn_shared_fence_handle`). The POD is copy-by-value; the only live resource
in it is an `fd`, which you transport to the peer out-of-band (`SCM_RIGHTS`
over a Unix socket) and write back into the received copy before
`OpenShared*`. Handle lifetime follows the two Windows regimes: legacy
`GetSharedHandle` values live and die with the exporting resource; NT-style
`CreateSharedHandle` values own an fd and must be released with
`dmn_shared_handle_close()`. `tests/shared_texture_test.cpp` and
`tests/fence_xapi_test.cpp` are complete producer/consumer examples.

## Environment variables

| Variable | Effect |
| --- | --- |
| `D3DMETAL_FRAMEWORK_PATH` | Path to `D3DMetal.framework` (or its binary). |
| `DMN_LOG` | Log threshold: `error`, `warn` (default), `info`, `debug`, `trace`, `quiet`. |
| `DMN_REG_<NAME>` | Overrides registry value `<NAME>` (uppercased, non-alphanumerics as `_`) at read time. |
| `DMN_VSYNC` | `0`/`1`: sets `displaySyncEnabled` on the presentation layer. |
| `DMN_RETINA` | Opt into the window's real backing scale (default is 1x, point == pixel). |
| `DMN_PRESENT_FALLBACK` | Present the last vended drawable from the host side if D3DMetal's own present path does not. |

## Tests

`meson test -C build` runs the suite: device bring-up, on-screen triangle and
cube demos, cross-process shared textures/buffers/fences/keyed mutexes in
both API directions, GPU waits on imported fences, event-driven waits,
shared heaps with placed resources, the exported swapchain backend,
lifecycle/fd-leak churn, and a multi-threaded stress test. The windowed demos
(`d3d11-triangle`, `d3d11-cube`, `d3d12-triangle --shared`,
`shared-compute-triangle` without `DMN_HEADLESS`) can also be run directly as
on-screen examples.

## License

MIT (see [LICENSE](LICENSE)). `D3DMetal.framework` and the dylibs bundled
with it are Apple's, licensed under the Game Porting Toolkit's own terms, and
are not part of this project. `tests/common/com.h` and the D3D11 triangle
demo are adapted from [dxvk] (zlib license).

[dxvk]: https://github.com/doitsujin/dxvk
