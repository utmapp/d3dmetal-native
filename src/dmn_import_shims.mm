/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * Public-API implementations of the private symbols D3DMetal.framework
 * imports, for builds that must not link private API (the Mac App Store
 * rejects a bundle whose D3DMetal references them).
 *
 * D3DMetal ships as a binary, so scripts/patch_d3dmetal_imports.py rewrites its import
 * table: each private import below is renamed to the dmn_* (or DMN*) symbol
 * here and bound to this library, and the CoreAnalytics load command becomes
 * the load command for this library. D3DMetal is only ever dlopen'd by this
 * library (dmn_init.cpp), so dyld resolves that load command to the
 * already-loaded image by install name. An unpatched framework never calls
 * any of this.
 *
 * Every replacement name must be no longer than the import it replaces: the
 * patcher renames in place. Keep the table in the patcher in sync with this
 * file; with --check-provider it refuses a library that does not export
 * every name it binds.
 *
 * What each private import did for D3DMetal, and what replaces it:
 *
 *   AnalyticsSendEventLazy (CoreAnalytics)   usage telemetry to Apple; never
 *       sent. The payload block is not invoked.
 *   MTLSetShaderCachePath,
 *   MTLMakeShaderCacheWritableByAllUsers     move Metal's own compiled-shader
 *       cache into D3DMetal's cache directory. No public equivalent; Metal
 *       keeps its default per-process cache, which still persists across
 *       runs. D3DMetal's own shader cache is unaffected.
 *   MTLTextureSwizzleChannelsToKey,
 *   MTLTextureSwizzleKeyToChannels           pack MTLTextureSwizzleChannels
 *       into the 32-bit key D3DMetal stores in its format table, and back.
 *       The key is the struct's four bytes, red in the low byte — verified
 *       against Metal for all 1296 channel combinations and checked again by
 *       tests/import_shims_test.mm.
 *   _dyld_get_image_uuid,
 *   dyld_image_header_containing_address     LC_UUID of D3DMetal and of
 *       libmetalirconverter, which key D3DMetal's shader cache. dladdr()
 *       finds the image and its load commands give the UUID.
 *   MTLAccelerationStructureAllocationDescriptor
 *                                            a plain descriptor D3DMetal
 *       creates, sets storageMode on, and hands back to Metal when it
 *       allocates a D3D12 acceleration structure. Metal reads it only through
 *       -storageMode and -forceResourceIndex, so this class implements the
 *       same properties, with the same defaults.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dlfcn.h>
#include <mach-o/loader.h>
#include <string.h>
#include <uuid/uuid.h>

#define DMN_SHIM_EXPORT extern "C" __attribute__((visibility("default")))

DMN_SHIM_EXPORT bool dmn_analytics_send(NSString* event,
                                        NSDictionary* (^payload)(void))
{
    (void)event;
    (void)payload;
    return false;
}

DMN_SHIM_EXPORT void dmn_metal_cache_set(NSString* path)
{
    (void)path;
}

DMN_SHIM_EXPORT void dmn_metal_cache_make_shared(void)
{
}

DMN_SHIM_EXPORT uint32_t dmn_swizzle_channels_to_key(
    MTLTextureSwizzleChannels channels)
{
    return (uint32_t)channels.red |
           (uint32_t)channels.green << 8 |
           (uint32_t)channels.blue << 16 |
           (uint32_t)channels.alpha << 24;
}

DMN_SHIM_EXPORT MTLTextureSwizzleChannels dmn_swizzle_key_to_channels(
    uint32_t key)
{
    return MTLTextureSwizzleChannelsMake(
        (MTLTextureSwizzle)(key & 0xff),
        (MTLTextureSwizzle)((key >> 8) & 0xff),
        (MTLTextureSwizzle)((key >> 16) & 0xff),
        (MTLTextureSwizzle)((key >> 24) & 0xff));
}

DMN_SHIM_EXPORT const struct mach_header* dmn_image_header_containing(
    const void* addr)
{
    Dl_info info;
    if (!addr || !dladdr(addr, &info))
        return nullptr;
    return static_cast<const struct mach_header*>(info.dli_fbase);
}

DMN_SHIM_EXPORT bool dmn_image_uuid(const struct mach_header* mh, uuid_t uuid)
{
    if (!mh || mh->magic != MH_MAGIC_64)
        return false;
    auto cmd = reinterpret_cast<const struct load_command*>(
        reinterpret_cast<const uint8_t*>(mh) + sizeof(struct mach_header_64));
    for (uint32_t i = 0; i < mh->ncmds; i++) {
        if (cmd->cmd == LC_UUID) {
            memcpy(uuid, reinterpret_cast<const struct uuid_command*>(cmd)->uuid,
                   sizeof(uuid_t));
            return true;
        }
        cmd = reinterpret_cast<const struct load_command*>(
            reinterpret_cast<const uint8_t*>(cmd) + cmd->cmdsize);
    }
    return false;
}

__attribute__((visibility("default")))
@interface DMNAccelerationStructureAllocationDescriptor : NSObject <NSCopying>
@property (nonatomic) MTLStorageMode storageMode;
@property (nonatomic) BOOL forceResourceIndex;
@property (nonatomic) NSUInteger resourceIndex;
@end

@implementation DMNAccelerationStructureAllocationDescriptor

/* Metal's descriptor starts out private, not shared. */
- (instancetype)init
{
    if ((self = [super init]))
        _storageMode = MTLStorageModePrivate;
    return self;
}

- (id)copyWithZone:(NSZone*)zone
{
    DMNAccelerationStructureAllocationDescriptor* copy =
        [[[self class] allocWithZone:zone] init];
    copy.storageMode = self.storageMode;
    copy.forceResourceIndex = self.forceResourceIndex;
    copy.resourceIndex = self.resourceIndex;
    return copy;
}

- (BOOL)isEqual:(id)other
{
    if (self == other)
        return YES;
    if (![other isKindOfClass:[DMNAccelerationStructureAllocationDescriptor class]])
        return NO;
    DMNAccelerationStructureAllocationDescriptor* o = other;
    return o.storageMode == self.storageMode &&
           o.forceResourceIndex == self.forceResourceIndex &&
           o.resourceIndex == self.resourceIndex;
}

- (NSUInteger)hash
{
    return self.storageMode ^ (NSUInteger)self.forceResourceIndex << 8 ^
           self.resourceIndex << 16;
}

@end
