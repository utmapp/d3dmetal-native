/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * d3dmetal-native: D3D11/D3D12 on macOS via Apple's D3DMetal.framework,
 * for native (non-Wine) applications.
 *
 * Usage model:
 *   1. Link against libd3dmetal-native.dylib ONLY. Do not link or dlopen
 *      D3DMetal.framework yourself; this library loads it and installs the
 *      GFXT host interface it requires.
 *   2. (Optional) call dmn_init() with options; otherwise the first D3D
 *      entry point (D3D11CreateDevice, CreateDXGIFactory1, ...) performs
 *      lazy default initialization.
 *   3. Create a dmn_window for your NSView or CAMetalLayer and pass
 *      dmn_window_get_hwnd() wherever a HWND is expected
 *      (IDXGIFactory2::CreateSwapChainForHwnd, D3D11CreateDeviceAndSwapChain).
 *
 * The D3D/DXGI entry points themselves are not declared here; supply your own
 * D3D headers (Wine/MinGW widl-generated ones). The symbols resolve from
 * libd3dmetal-native at link time. TUs that include d3d12.h must be compiled
 * with -DWIDL_EXPLICIT_AGGREGATE_RETURNS=1 to match D3DMetal's ABI for the
 * D3D12 methods that return aggregates by value (GetDesc,
 * GetCPUDescriptorHandleForHeapStart, ...).
 *
 * This header is pure C with no Windows or Cocoa type dependencies and is
 * safe to include from C, C++, Objective-C, and Objective-C++.
 *
 * IMPORTANT: D3DMetal.framework is x86_64-only. The entire process must be
 * x86_64 (build with -arch x86_64; runs under Rosetta 2 on Apple Silicon).
 */

#ifndef D3DMETAL_NATIVE_H
#define D3DMETAL_NATIVE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DMN_VERSION_MAJOR 0
#define DMN_VERSION_MINOR 1
#define DMN_VERSION_PATCH 0

/* Typed Cocoa handles when compiled as Objective-C; opaque otherwise. */
#ifdef __OBJC__
@class NSView;
@class CAMetalLayer;
typedef NSView*       dmn_nsview_t;
typedef CAMetalLayer* dmn_metal_layer_t;
#else
typedef void* dmn_nsview_t;
typedef void* dmn_metal_layer_t;
#endif

typedef enum dmn_result {
    DMN_SUCCESS                   =  0,
    DMN_ERROR_FRAMEWORK_NOT_FOUND = -1, /* no D3DMetal binary at any search path */
    DMN_ERROR_SYMBOL_NOT_FOUND    = -2, /* dlsym(GFXT_Initialize) failed */
    DMN_ERROR_NOT_INITIALIZED     = -3,
    DMN_ERROR_INVALID_ARGUMENT    = -4,
    DMN_ERROR_UNSUPPORTED         = -5,
    DMN_ERROR_OUT_OF_MEMORY       = -6,
} dmn_result;

typedef enum dmn_log_level {
    DMN_LOG_ERROR = 0,
    DMN_LOG_WARN  = 1,
    DMN_LOG_INFO  = 2,
    DMN_LOG_DEBUG = 3,
    DMN_LOG_TRACE = 4, /* every GFXT host call logs at this level */
} dmn_log_level;

typedef struct dmn_options {
    /* Must be set to sizeof(dmn_options); allows forward-compatible growth. */
    size_t struct_size;

    /* Explicit path to the D3DMetal.framework directory or the Mach-O binary
     * inside it. NULL selects the default search order:
     *   $D3DMETAL_FRAMEWORK_PATH environment variable
     *   <dir of libd3dmetal-native.dylib>/D3DMetal.framework
     *   <dir of libd3dmetal-native.dylib>/../Frameworks/D3DMetal.framework
     *   compiled-in development default. */
    const char* framework_path;

    /* Optional log sink; NULL logs to stderr. The threshold is controlled by
     * the DMN_LOG environment variable (error|warn|info|debug|trace),
     * regardless of sink. */
    void (*log_callback)(dmn_log_level level, const char* message, void* ctx);
    void* log_ctx;
} dmn_options;

/* Load D3DMetal.framework and install the GFXT host interface via its
 * GFXT_Initialize entry point. Idempotent: subsequent calls return
 * DMN_SUCCESS and ignore the options. Thread-safe. options may be NULL.
 * There is no shutdown; D3DMetal has no teardown path, so initialization
 * lasts for the lifetime of the process. */
dmn_result dmn_init(const dmn_options* options);

bool dmn_is_initialized(void);

/* Resolved absolute path of the loaded framework binary; NULL before init. */
const char* dmn_framework_path(void);

/* == Windows / pseudo-HWNDs ============================================== */

typedef struct dmn_window* dmn_window_t;

/* Create a window object backed by an existing NSView. d3dmetal-native
 * creates and owns a CAMetalLayer and installs it as the view's backing
 * layer (sets wantsLayer). MUST be called on the main thread (AppKit rule).
 * The view is retained until dmn_window_destroy. Returns NULL on error. */
dmn_window_t dmn_window_create_for_view(dmn_nsview_t view);

/* Create a window object backed by a caller-owned CAMetalLayer (retained
 * until dmn_window_destroy). No view is involved; usable offscreen or when
 * the caller manages the layer hierarchy itself. Callable from any thread. */
dmn_window_t dmn_window_create_for_layer(dmn_metal_layer_t layer);

/* The value to pass anywhere a HWND is expected. Stable for the lifetime of
 * the window. Always a small (< 2^32) non-NULL pseudo-handle, never a raw
 * pointer — safe against 32-bit handle truncation inside D3DMetal. */
void* dmn_window_get_hwnd(dmn_window_t window);

/* Reverse lookup; NULL if hwnd is not a live dmn pseudo-handle. */
dmn_window_t dmn_window_from_hwnd(void* hwnd);

/* The CAMetalLayer in use (borrowed reference). */
dmn_metal_layer_t dmn_window_get_metal_layer(dmn_window_t window);

/* Inform d3dmetal-native of the desired backing size in PIXELS and the
 * backing scale factor (applied to layer.contentsScale). Call before
 * swapchain creation and on view resize, in addition to
 * IDXGISwapChain::ResizeBuffers (which drives the GFXT ResizeBacking path).
 * Callable from any thread. */
void dmn_window_set_size(dmn_window_t window,
                         uint32_t pixel_width, uint32_t pixel_height,
                         double contents_scale);

/* Destroy the window object. A library-created CAMetalLayer is detached from
 * the view on the main thread. Destroy swapchains referencing this hwnd
 * first; afterwards GFXT calls with the stale hwnd fail gracefully. */
void dmn_window_destroy(dmn_window_t window);

/* == Win32-style event interop =========================================== */
/* HANDLE values returned through D3D APIs (e.g.
 * IDXGISwapChain2::GetFrameLatencyWaitableObject) are created by this
 * library's GFXT event implementation and can be waited on here. */

#define DMN_WAIT_INFINITE UINT64_MAX

typedef enum dmn_wait_status {
    DMN_WAIT_SIGNALED = 0,
    DMN_WAIT_TIMEOUT  = 1,
    DMN_WAIT_FAILED   = -1, /* not a dmn event/semaphore handle */
} dmn_wait_status;

dmn_wait_status dmn_event_wait(void* handle, uint64_t timeout_ns);

/* Events are kqueue-backed: each event owns a kqueue whose single
 * EVFILT_USER registration IS the signaled state, so one event is
 * simultaneously waitable (dmn_event_wait, a blocking kevent) and
 * pollable (dmn_event_dup_fd).  Cost: one fd per live event — raise
 * RLIMIT_NOFILE if you host very many (macOS's default soft limit is
 * low).  Semaphores are libdispatch objects, fd-free. */

/* Create a standalone event HANDLE of the same kind D3DMetal's GFXT event
 * interface vends, so it is accepted anywhere a D3D API takes a Win32 event
 * HANDLE (ID3D11Fence/ID3D12Fence::SetEventOnCompletion, ...) and D3DMetal's
 * own SetEvent will signal it. manual_reset/initial_state follow Win32
 * CreateEvent semantics (non-zero = true). Waitable with dmn_event_wait();
 * release with dmn_event_close(). NULL on failure (fd exhaustion). */
void* dmn_event_create(int manual_reset, int initial_state);

/* Win32 SetEvent / ResetEvent analogs for handles from dmn_event_create (or
 * any event HANDLE surfaced through the D3D APIs). No-ops with a warning on
 * non-event handles. */
void dmn_event_signal(void* handle);
void dmn_event_clear(void* handle);

/* Destroy the handle. Waiters must not be blocked on it, and D3DMetal must
 * no longer hold it (Duplicate'd handles inside D3DMetal are independent).
 * fds returned by dmn_event_dup_fd survive the close. */
void dmn_event_close(void* handle);

/* Pollable view of an event: returns a NEW fd (caller owns; close(2) it)
 * that polls readable (POLLIN) while the event is signaled and unreadable
 * while it is clear. Backed by a pipe whose readable state mirrors the
 * event's kqueue trigger (macOS cannot pass kqueue fds across processes),
 * shared across DuplicateEvent, so every fd returned for a handle — or any
 * duplicate of it — tracks later signal/clear transitions, including
 * D3DMetal's internal SetEvent on fence completion. Returns -1 on failure
 * or if handle is not an event. Auto-reset caveat: a dmn_event_wait
 * consumer that wins the race clears readability too (Win32 semantics). */
int dmn_event_dup_fd(void* handle);

/* Override what D3DMetal sees as the running executable's path (its per-app
 * profile matcher keys on the basename). Must be set before the first D3D
 * device/factory creation to be effective. NULL restores the real process
 * path. The string is copied. Useful for embedders hosting another
 * program's D3D workload (VM display servers, remoting hosts). */
dmn_result dmn_set_executable_path(const char* path);

/* == Configuration ======================================================= */
/* D3DMetal reads settings through the GFXT registry interface, backed here
 * by an in-memory store. Values may be pre-seeded before or after init.
 * Example: dmn_registry_set_u32(DMN_HKEY_CURRENT_USER,
 *                               "Software\\Apple\\D3DMetal", "logLevel", 3);
 * The environment variable DMN_REG_<NAME>=<value> overrides any value name
 * <NAME> (case-insensitive) at read time. */

typedef enum dmn_registry_root {
    DMN_HKEY_CLASSES_ROOT  = 0,
    DMN_HKEY_CURRENT_USER  = 1,
    DMN_HKEY_LOCAL_MACHINE = 2,
    DMN_HKEY_USERS         = 3,
} dmn_registry_root;

dmn_result dmn_registry_set_u32(dmn_registry_root root, const char* subkey,
                                const char* name, uint32_t value);
dmn_result dmn_registry_set_u64(dmn_registry_root root, const char* subkey,
                                const char* name, uint64_t value);
dmn_result dmn_registry_set_string(dmn_registry_root root, const char* subkey,
                                   const char* name, const char* value);

/* == Cross-process shared textures & fences ============================= */
/*
 * The bundled D3DMetal v3.0 stubs out every resource/fence sharing entry
 * point (CreateSharedHandle logs "faking with passthrough", OpenSharedHandle
 * returns E_NOTIMPL, the D3D11 OpenSharedResource* / GetSharedHandle family
 * no-op). d3dmetal-native re-implements real cross-process sharing on top of
 * Metal StorageModeShared buffers backed by anonymous shared memory, exposed
 * ENTIRELY through the standard D3D APIs via vtable interception — there are
 * no d3dmetal-native sharing entry points to call:
 *
 *   Producer: CreateTexture2D/CreateTexture2D1 (D3D11_RESOURCE_MISC_SHARED*),
 *     CreateBuffer (MISC_SHARED), CreateCommittedResource / CreatePlacedResource
 *     and their 1/2/3 variants (D3D12_HEAP_FLAG_SHARED), or CreateFence
 *     (SHARED). Export with IDXGIResource::GetSharedHandle,
 *     IDXGIResource1/ID3D11Fence::CreateSharedHandle, or
 *     ID3D12Device::CreateSharedHandle — each returns a HANDLE that is really a
 *     pointer to one of the PODs below. (Named handles — the *ByName openers —
 *     are unsupported; ship the POD + fd instead.)
 *   Consumer: ID3D11Device::OpenSharedResource / OpenSharedResource1 or
 *     ID3D12Device::OpenSharedHandle reconstruct the texture/buffer, or vend a
 *     fence, dispatching on the POD magic. Import is API-agnostic: a D3D11
 *     producer's texture opens fine in a D3D12 consumer and vice versa.
 *   Synchronization: an imported fence supports GPU-side ID3D12CommandQueue /
 *     ID3D11DeviceContext4 Wait and Signal (bridged internally), and a texture
 *     created with D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX vends a working
 *     cross-process IDXGIKeyedMutex via QueryInterface.
 *
 * The POD is copy-by-value; the ONLY live resource it references is the fd,
 * which the caller transports out-of-band (SCM_RIGHTS over a unix socket) and
 * writes back into the receiver's copy's `fd` field before OpenShared*: the
 * fd is the token, the struct is the layout.
 *
 * Handle lifetime follows the two Windows regimes:
 *
 *   Legacy handles — IDXGIResource::GetSharedHandle. No lifecycle of their
 *   own (MSDN: "this handle is not an NT handle", never closed). The HANDLE
 *   points at memory owned by the exporting resource: it is valid while that
 *   resource is alive and invalid after its final Release. Repeated calls
 *   return the same HANDLE. Do NOT pass these to dmn_shared_handle_close().
 *
 *   NT-style handles — IDXGIResource1::CreateSharedHandle,
 *   ID3D11Fence::CreateSharedHandle, ID3D12Device::CreateSharedHandle. Each
 *   call returns a NEW handle that owns its own duplicate of the fd and
 *   remains valid after the resource is released (the Windows rule: the NT
 *   handle references the allocation until CloseHandle). The consumer MUST
 *   call dmn_shared_handle_close() exactly once per handle or the POD and
 *   its fd leak. (Divergence: Windows fails a second CreateSharedHandle on
 *   the same resource; here each call succeeds with a fresh handle.)
 *
 *   Either way, an OPENED resource may be re-exported (GetSharedHandle /
 *   CreateSharedHandle on it) — every registration owns its own duplicate of
 *   the fd, so handles exported from an opened resource survive the creator,
 *   matching MSDN's rule that any resource object referring to the memory
 *   extends the sharing's lifetime. (Note: D3D12 has no legacy regime at all
 *   on Windows — its only sharing entry points are the NT-shaped pair.)
 */

/* Close a handle returned by the CreateSharedHandle family (the CloseHandle
 * analog; closes the handle's fd and frees the POD). Returns
 * DMN_ERROR_INVALID_ARGUMENT for NULL, a legacy GetSharedHandle value, an
 * already-closed handle, or any pointer this library did not vend — nothing
 * is touched in that case. */
dmn_result dmn_shared_handle_close(void* handle);

#define DMN_SHARED_TEXTURE_MAGIC 0x58544D44u /* 'DMTX' */
#define DMN_SHARED_FENCE_MAGIC   0x4E464D44u /* 'DMFN' */
#define DMN_SHARED_HANDLE_VERSION 1u

typedef struct dmn_shared_texture_handle {
    uint32_t magic;         /* DMN_SHARED_TEXTURE_MAGIC */
    uint32_t version;       /* DMN_SHARED_HANDLE_VERSION */
    int32_t  fd;            /* process-local; send via SCM_RIGHTS, then patch */
    uint32_t width, height;
    uint32_t dxgi_format;   /* DXGI_FORMAT */
    uint32_t mip_levels, array_size, sample_count;
    uint32_t bind_flags, misc_flags, cpu_access;
    uint64_t stride;        /* bytesPerRow, byte-exact for the consumer */
    uint64_t size;          /* logical stride*height (NOT page-padded) */
} dmn_shared_texture_handle;

typedef struct dmn_shared_fence_handle {
    uint32_t magic;         /* DMN_SHARED_FENCE_MAGIC */
    uint32_t version;       /* DMN_SHARED_HANDLE_VERSION */
    int32_t  fd;            /* companion value buffer; SCM_RIGHTS + patch.
                               The fence's uint64 value slot is at offset 0. */
    uint32_t flags;         /* D3D11_FENCE_FLAG / D3D12_FENCE_FLAGS */
    uint64_t initial_value;
} dmn_shared_fence_handle;

/* == Cross-process shared buffers ======================================== */
/* Like shared textures, but for D3D buffers created with the shared flag
 * (D3D11 D3D11_RESOURCE_MISC_SHARED, D3D12 D3D12_HEAP_FLAG_SHARED). */

#define DMN_SHARED_BUFFER_MAGIC 0x42544D44u /* 'DMTB' */

typedef struct dmn_shared_buffer_handle {
    uint32_t magic;         /* DMN_SHARED_BUFFER_MAGIC */
    uint32_t version;       /* DMN_SHARED_HANDLE_VERSION */
    int32_t  fd;            /* process-local; send via SCM_RIGHTS, then patch */
    uint32_t bind_flags, misc_flags, cpu_access;
    uint64_t size;          /* logical byte length */
} dmn_shared_buffer_handle;

#ifdef __cplusplus
}
#endif

#endif /* D3DMETAL_NATIVE_H */
