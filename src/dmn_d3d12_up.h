/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Hand-declared ID3D12Device2..ID3D12Device10 (the vendored d3d12.h stops at
 * ID3D12Device1). D3DMetal answers QueryInterface for all of them, and apps
 * bring their own headers, so the shared-resource create variants they add
 * (CreateCommittedResource1/2/3, CreateHeap1, CreatePlacedResource1/2,
 * CreateCommandQueue1) must be hookable. These interface layouts are frozen
 * published ABIs; method ORDER is load-bearing (it defines the vtable slots
 * dmn_vindex computes), signatures matter only for the methods we hook or
 * call. Parameters whose types the vendored headers lack are declared as
 * ABI-equivalent opaque pointers/UINTs.
 *
 * The GetResourceAllocationInfo1/2 slots return an aggregate by value; they
 * are declared in the explicit-return-pointer form this project builds with
 * (WIDL_EXPLICIT_AGGREGATE_RETURNS) and must not be called through these
 * declarations anyway — they exist only to hold their vtable slot.
 *
 * Internal-only header (never installed).
 */

#pragma once

#include <d3d12.h>

/* == Types the vendored headers lack ====================================== */

typedef struct D3D12_MIP_REGION {
    UINT Width;
    UINT Height;
    UINT Depth;
} D3D12_MIP_REGION;

/* D3D12_RESOURCE_DESC + SamplerFeedbackMipRegion. */
typedef struct D3D12_RESOURCE_DESC1 {
    D3D12_RESOURCE_DIMENSION Dimension;
    UINT64 Alignment;
    UINT64 Width;
    UINT Height;
    UINT16 DepthOrArraySize;
    UINT16 MipLevels;
    DXGI_FORMAT Format;
    DXGI_SAMPLE_DESC SampleDesc;
    D3D12_TEXTURE_LAYOUT Layout;
    D3D12_RESOURCE_FLAGS Flags;
    D3D12_MIP_REGION SamplerFeedbackMipRegion;
} D3D12_RESOURCE_DESC1;

typedef struct D3D12_RESOURCE_ALLOCATION_INFO1 {
    UINT64 Offset;
    UINT64 Alignment;
    UINT64 SizeInBytes;
} D3D12_RESOURCE_ALLOCATION_INFO1;

/* 4-byte enums we only forward; ABI-equivalent to UINT. */
typedef UINT DMN_D3D12_BARRIER_LAYOUT;

/* ID3D12ProtectedResourceSession is only ever forwarded. */
typedef IUnknown DMN_ID3D12ProtectedResourceSession;

/* == ID3D12Device2..10 ===================================================== */

MIDL_INTERFACE("30baa41e-b15b-475c-a0bb-1af5c5b64328")
ID3D12Device2 : public ID3D12Device1
{
    virtual HRESULT STDMETHODCALLTYPE CreatePipelineState(
        const void *desc, REFIID iid, void **state) = 0;
};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(ID3D12Device2, 0x30baa41e, 0xb15b, 0x475c, 0xa0,0xbb, 0x1a,0xf5,0xc5,0xb6,0x43,0x28)
#endif

MIDL_INTERFACE("81dadc15-2bad-4392-93c5-101345c4aa98")
ID3D12Device3 : public ID3D12Device2
{
    virtual HRESULT STDMETHODCALLTYPE OpenExistingHeapFromAddress(
        const void *address, REFIID iid, void **heap) = 0;
    virtual HRESULT STDMETHODCALLTYPE OpenExistingHeapFromFileMapping(
        HANDLE file_mapping, REFIID iid, void **heap) = 0;
    virtual HRESULT STDMETHODCALLTYPE EnqueueMakeResident(
        UINT flags, UINT num_objects, ID3D12Pageable *const *objects,
        ID3D12Fence *fence, UINT64 fence_value) = 0;
};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(ID3D12Device3, 0x81dadc15, 0x2bad, 0x4392, 0x93,0xc5, 0x10,0x13,0x45,0xc4,0xaa,0x98)
#endif

MIDL_INTERFACE("e865df17-a9ee-46f9-a463-3098315aa2e5")
ID3D12Device4 : public ID3D12Device3
{
    virtual HRESULT STDMETHODCALLTYPE CreateCommandList1(
        UINT node_mask, D3D12_COMMAND_LIST_TYPE type, UINT flags,
        REFIID iid, void **list) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateProtectedResourceSession(
        const void *desc, REFIID iid, void **session) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateCommittedResource1(
        const D3D12_HEAP_PROPERTIES *heap_properties, D3D12_HEAP_FLAGS heap_flags,
        const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES initial_state,
        const D3D12_CLEAR_VALUE *optimized_clear_value,
        DMN_ID3D12ProtectedResourceSession *protected_session,
        REFIID iid, void **resource) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateHeap1(
        const D3D12_HEAP_DESC *desc,
        DMN_ID3D12ProtectedResourceSession *protected_session,
        REFIID iid, void **heap) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateReservedResource1(
        const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES initial_state,
        const D3D12_CLEAR_VALUE *optimized_clear_value,
        DMN_ID3D12ProtectedResourceSession *protected_session,
        REFIID iid, void **resource) = 0;
    /* Aggregate-by-value return; slot placeholder only — do not call. */
    virtual D3D12_RESOURCE_ALLOCATION_INFO* STDMETHODCALLTYPE GetResourceAllocationInfo1(
        D3D12_RESOURCE_ALLOCATION_INFO *ret, UINT visible_mask,
        UINT num_resource_descs, const D3D12_RESOURCE_DESC *descs,
        D3D12_RESOURCE_ALLOCATION_INFO1 *info1) = 0;
};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(ID3D12Device4, 0xe865df17, 0xa9ee, 0x46f9, 0xa4,0x63, 0x30,0x98,0x31,0x5a,0xa2,0xe5)
#endif

MIDL_INTERFACE("8b4f173b-2fea-4b80-8f58-4307191ab95d")
ID3D12Device5 : public ID3D12Device4
{
    virtual HRESULT STDMETHODCALLTYPE CreateLifetimeTracker(
        void *owner, REFIID iid, void **tracker) = 0;
    virtual void STDMETHODCALLTYPE RemoveDevice() = 0;
    virtual HRESULT STDMETHODCALLTYPE EnumerateMetaCommands(
        UINT *num_meta_commands, void *descs) = 0;
    virtual HRESULT STDMETHODCALLTYPE EnumerateMetaCommandParameters(
        REFGUID command_id, UINT stage, UINT *total_structure_size,
        UINT *parameter_count, void *parameter_descs) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateMetaCommand(
        REFGUID command_id, UINT node_mask, const void *creation_parameters,
        SIZE_T creation_parameters_size, REFIID iid, void **meta_command) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateStateObject(
        const void *desc, REFIID iid, void **state_object) = 0;
    virtual void STDMETHODCALLTYPE GetRaytracingAccelerationStructurePrebuildInfo(
        const void *desc, void *info) = 0;
    virtual UINT STDMETHODCALLTYPE CheckDriverMatchingIdentifier(
        UINT serialized_data_type, void *identifier_to_check) = 0;
};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(ID3D12Device5, 0x8b4f173b, 0x2fea, 0x4b80, 0x8f,0x58, 0x43,0x07,0x19,0x1a,0xb9,0x5d)
#endif

MIDL_INTERFACE("c70b221b-40e4-4a17-89af-025a0727a6dc")
ID3D12Device6 : public ID3D12Device5
{
    virtual HRESULT STDMETHODCALLTYPE SetBackgroundProcessingMode(
        UINT mode, UINT measurements_action, HANDLE event_to_signal,
        BOOL *further_measurements_desired) = 0;
};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(ID3D12Device6, 0xc70b221b, 0x40e4, 0x4a17, 0x89,0xaf, 0x02,0x5a,0x07,0x27,0xa6,0xdc)
#endif

MIDL_INTERFACE("5c014b53-68a1-4b9b-8bd1-dd6046b9358b")
ID3D12Device7 : public ID3D12Device6
{
    virtual HRESULT STDMETHODCALLTYPE AddToStateObject(
        const void *addition, void *state_object_to_grow_from,
        REFIID iid, void **new_state_object) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateProtectedResourceSession1(
        const void *desc, REFIID iid, void **session) = 0;
};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(ID3D12Device7, 0x5c014b53, 0x68a1, 0x4b9b, 0x8b,0xd1, 0xdd,0x60,0x46,0xb9,0x35,0x8b)
#endif

MIDL_INTERFACE("9218e6bb-f944-4f7e-a75c-b1b2c7b701f3")
ID3D12Device8 : public ID3D12Device7
{
    /* Aggregate-by-value return; slot placeholder only — do not call. */
    virtual D3D12_RESOURCE_ALLOCATION_INFO* STDMETHODCALLTYPE GetResourceAllocationInfo2(
        D3D12_RESOURCE_ALLOCATION_INFO *ret, UINT visible_mask,
        UINT num_resource_descs, const D3D12_RESOURCE_DESC1 *descs,
        D3D12_RESOURCE_ALLOCATION_INFO1 *info1) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateCommittedResource2(
        const D3D12_HEAP_PROPERTIES *heap_properties, D3D12_HEAP_FLAGS heap_flags,
        const D3D12_RESOURCE_DESC1 *desc, D3D12_RESOURCE_STATES initial_state,
        const D3D12_CLEAR_VALUE *optimized_clear_value,
        DMN_ID3D12ProtectedResourceSession *protected_session,
        REFIID iid, void **resource) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreatePlacedResource1(
        ID3D12Heap *heap, UINT64 heap_offset, const D3D12_RESOURCE_DESC1 *desc,
        D3D12_RESOURCE_STATES initial_state,
        const D3D12_CLEAR_VALUE *optimized_clear_value,
        REFIID iid, void **resource) = 0;
    virtual void STDMETHODCALLTYPE CreateSamplerFeedbackUnorderedAccessView(
        ID3D12Resource *targeted_resource, ID3D12Resource *feedback_resource,
        D3D12_CPU_DESCRIPTOR_HANDLE descriptor) = 0;
    virtual void STDMETHODCALLTYPE GetCopyableFootprints1(
        const D3D12_RESOURCE_DESC1 *desc, UINT first_subresource,
        UINT num_subresources, UINT64 base_offset,
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT *layouts, UINT *num_rows,
        UINT64 *row_size_in_bytes, UINT64 *total_bytes) = 0;
};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(ID3D12Device8, 0x9218e6bb, 0xf944, 0x4f7e, 0xa7,0x5c, 0xb1,0xb2,0xc7,0xb7,0x01,0xf3)
#endif

MIDL_INTERFACE("4c80e962-f032-4f60-bc9e-ebc2cfa1d83c")
ID3D12Device9 : public ID3D12Device8
{
    virtual HRESULT STDMETHODCALLTYPE CreateShaderCacheSession(
        const void *desc, REFIID iid, void **session) = 0;
    virtual HRESULT STDMETHODCALLTYPE ShaderCacheControl(
        UINT kinds, UINT control) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateCommandQueue1(
        const D3D12_COMMAND_QUEUE_DESC *desc, REFIID creator_id,
        REFIID iid, void **queue) = 0;
};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(ID3D12Device9, 0x4c80e962, 0xf032, 0x4f60, 0xbc,0x9e, 0xeb,0xc2,0xcf,0xa1,0xd8,0x3c)
#endif

MIDL_INTERFACE("517f8718-aa66-49f9-b02b-a7ab89c06031")
ID3D12Device10 : public ID3D12Device9
{
    virtual HRESULT STDMETHODCALLTYPE CreateCommittedResource3(
        const D3D12_HEAP_PROPERTIES *heap_properties, D3D12_HEAP_FLAGS heap_flags,
        const D3D12_RESOURCE_DESC1 *desc, DMN_D3D12_BARRIER_LAYOUT initial_layout,
        const D3D12_CLEAR_VALUE *optimized_clear_value,
        DMN_ID3D12ProtectedResourceSession *protected_session,
        UINT32 num_castable_formats, const DXGI_FORMAT *castable_formats,
        REFIID iid, void **resource) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreatePlacedResource2(
        ID3D12Heap *heap, UINT64 heap_offset, const D3D12_RESOURCE_DESC1 *desc,
        DMN_D3D12_BARRIER_LAYOUT initial_layout,
        const D3D12_CLEAR_VALUE *optimized_clear_value,
        UINT32 num_castable_formats, const DXGI_FORMAT *castable_formats,
        REFIID iid, void **resource) = 0;
    virtual HRESULT STDMETHODCALLTYPE CreateReservedResource2(
        const D3D12_RESOURCE_DESC *desc, DMN_D3D12_BARRIER_LAYOUT initial_layout,
        const D3D12_CLEAR_VALUE *optimized_clear_value,
        DMN_ID3D12ProtectedResourceSession *protected_session,
        UINT32 num_castable_formats, const DXGI_FORMAT *castable_formats,
        REFIID iid, void **resource) = 0;
};
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(ID3D12Device10, 0x517f8718, 0xaa66, 0x49f9, 0xb0,0x2b, 0xa7,0xab,0x89,0xc0,0x60,0x31)
#endif
