/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * DmnGFXT — the GFXTOSInterface root D3DMetal receives via
 * GFXT_Initialize. Sub-interfaces are process-lifetime singletons
 * (matching libd3dshared's behavior); swapchain interfaces are created
 * per call and owned by D3DMetal.
 */

#include "dmn_gfxt.h"
#include "dmn_log.h"

DmnGFXT::DmnGFXT()
    : events_(new DmnGFXTEvent()),
      monitor_(new DmnGFXTMonitor()),
      registry_(new DmnGFXTRegistry()),
      adapters_(new DmnGFXTAdapter()),
      paths_(new DmnGFXTPath()),
      allocations_(new DmnGFXTAllocation()),
      libraries_(new DmnGFXTLibrary()) {}

DmnGFXT::~DmnGFXT() = default;

GFXTMonitorInterface*
DmnGFXT::CreateMonitorInterface(const GFXTInterfaceVersion& v) {
    DMN_INFO("CreateMonitorInterface(version=%u)", v.version);
    return monitor_.get();
}

GFXTRegistryInterface*
DmnGFXT::CreateRegistryInterface(const GFXTInterfaceVersion& v) {
    DMN_INFO("CreateRegistryInterface(version=%u)", v.version);
    return registry_.get();
}

GFXTEventInterface*
DmnGFXT::CreateEventInterface(const GFXTInterfaceVersion& v) {
    DMN_INFO("CreateEventInterface(version=%u)", v.version);
    return events_.get();
}

GFXTSwapchainInterface*
DmnGFXT::CreateSwapchainInterface(const GFXTInterfaceVersion& v,
                                  void* mtlDevice) {
    DMN_INFO("CreateSwapchainInterface(version=%u, mtlDevice=%p)",
             v.version, mtlDevice);
    return new DmnGFXTSwapchain(mtlDevice);
}

GFXTAdapterInterface*
DmnGFXT::CreateAdapterInterface(const GFXTInterfaceVersion& v) {
    DMN_INFO("CreateAdapterInterface(version=%u)", v.version);
    return adapters_.get();
}

GFXTPathInterface*
DmnGFXT::CreatePathInterface(const GFXTInterfaceVersion& v) {
    DMN_INFO("CreatePathInterface(version=%u)", v.version);
    return paths_.get();
}

GFXTAllocationInterface*
DmnGFXT::CreateAllocationInterface(const GFXTInterfaceVersion& v) {
    DMN_INFO("CreateAllocationInterface(version=%u)", v.version);
    return allocations_.get();
}

GFXTLibraryInterface*
DmnGFXT::CreateLibraryInterface(const GFXTInterfaceVersion& v) {
    DMN_INFO("CreateLibraryInterface(version=%u)", v.version);
    return libraries_.get();
}
