project('d3dmetal-native', ['c', 'cpp', 'objcpp'],
  version: '0.1.0',
  license: 'MIT',
  meson_version: '>= 0.60',
  default_options: [
    'cpp_std=c++17',
    'warning_level=2',
    'b_ndebug=if-release',
  ])

if host_machine.system() != 'darwin'
  error('d3dmetal-native targets macOS only')
endif
if host_machine.cpu_family() != 'x86_64'
  error('D3DMetal.framework is x86_64-only; configure with ' +
        '--cross-file build-macos-x86_64.txt')
endif

add_project_arguments('-std=gnu++17', '-fno-objc-arc', language: 'objcpp')

dev_fw = get_option('dev_framework_path')
if dev_fw == ''
  dev_fw = meson.project_source_root() / 'D3DMetal.framework/Versions/A/D3DMetal'
endif
add_project_arguments('-DDMN_DEV_FRAMEWORK_PATH="' + dev_fw + '"',
                      language: ['cpp', 'objcpp'])

# D3DMetal is built from the Wine/MinGW D3D headers, so the D3D12 methods that
# return a small aggregate by value (ID3D12DescriptorHeap::GetCPU/GPUDescriptor-
# HandleForHeapStart, ID3D12Resource::GetDesc, GetResourceAllocationInfo, ...)
# use the explicit-return-pointer ABI: the vtable slot takes a hidden struct
# pointer. Any TU that includes d3d12.h and calls those methods MUST agree, or
# the by-value form omits the pointer and D3DMetal writes through a garbage
# register (SIGSEGV). Only d3d12.h consults this macro; D3D11/DXGI are
# unaffected. Set it project-wide so no call site can regress.
add_project_arguments('-DWIDL_EXPLICIT_AGGREGATE_RETURNS=1',
                      language: ['cpp', 'objcpp'])

apple_deps = dependency('appleframeworks',
  modules: ['Foundation', 'AppKit', 'Metal', 'QuartzCore', 'CoreGraphics'])

dmn_public_inc = include_directories('include')
dmn_private_inc = include_directories('src')
# The COM-hook TU (dmn_com_hooks.cpp) compiles against the vendored DirectX /
# Windows headers to patch real interface vtables.
dmn_native_inc = include_directories(
  'include/native/windows',
  'include/native/directx',
)

dmn_sources = files(
  'src/dmn_log.cpp',
  'src/dmn_init.cpp',
  'src/dmn_gfxt_os.cpp',
  'src/dmn_gfxt_event.cpp',
  'src/dmn_gfxt_registry.cpp',
  'src/dmn_gfxt_path.cpp',
  'src/dmn_gfxt_allocation.cpp',
  'src/dmn_gfxt_library.cpp',
  'src/dmn_gfxt_monitor.cpp',
  'src/dmn_gfxt_adapter.mm',
  'src/dmn_window.mm',
  'src/dmn_gfxt_swapchain.mm',
  'src/dmn_share_metal.mm',
  'src/dmn_fence.cpp',
  'src/dmn_fence_d3d.cpp',
  'src/dmn_kmtx.cpp',
  'src/dmn_com_hooks.cpp',
)

exports_list = meson.current_source_dir() / 'src/exports.symbols'

dmn_lib = library('d3dmetal-native', dmn_sources,
  include_directories: [dmn_public_inc, dmn_private_inc, dmn_native_inc],
  dependencies: [apple_deps],
  link_args: ['-Wl,-exported_symbols_list,' + exports_list],
  link_depends: files('src/exports.symbols'),
  install: true)

# The vendored Wine/MinGW headers under include/native are internal-only (used
# by the library and tests); consumers bring their own D3D headers.
install_headers('include/d3dmetal_native.h', subdir: 'd3dmetal-native')

pkg = import('pkgconfig')
pkg.generate(dmn_lib,
  name: 'd3dmetal-native',
  description: 'D3D11/D3D12 for native macOS apps via D3DMetal.framework',
  subdirs: ['d3dmetal-native'])

if get_option('tests').allowed()
  subdir('tests')
endif
