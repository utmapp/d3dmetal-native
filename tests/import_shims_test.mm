/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * import_shims_test: each public-API replacement in dmn_import_shims.mm must
 * behave exactly like the private symbol a patched D3DMetal no longer imports.
 * The test may call the private originals (tests are not shipped), so every
 * comparison is against the real thing:
 *
 *   swizzle pair   all 6^4 channel combinations, both directions, plus
 *                  out-of-range bytes for KeyToChannels.
 *   dyld pair      header and LC_UUID for D3DMetal, libmetalirconverter and
 *                  this executable, interior addresses, and a heap address.
 *   AS descriptor  Metal allocates an acceleration structure from our class
 *                  exactly as it does from MTLAccelerationStructureAllocation-
 *                  Descriptor, with the same defaults; and a D3D12
 *                  acceleration-structure create still works (reporting
 *                  whether D3DMetal went through our class).
 *
 * Prints "IMPORT-SHIMS: PASS" and exits 0 on success.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <dlfcn.h>
#include <libgen.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <uuid/uuid.h>

#include <string>

#include "d3dmetal_native.h"

extern "C" {
bool dmn_analytics_send(NSString* event, NSDictionary* (^payload)(void));
uint32_t dmn_swizzle_channels_to_key(MTLTextureSwizzleChannels channels);
MTLTextureSwizzleChannels dmn_swizzle_key_to_channels(uint32_t key);
const struct mach_header* dmn_image_header_containing(const void* addr);
bool dmn_image_uuid(const struct mach_header* mh, uuid_t uuid);
int import_shims_create_as(unsigned long long* gpu_va);
}

#define T_TAG "IMPORT-SHIMS"

static int g_fail = 0;

#define EXPECT(cond, ...)                                                    \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, T_TAG ": FAIL " __VA_ARGS__);                    \
            fputc('\n', stderr);                                             \
            g_fail++;                                                        \
        }                                                                    \
    } while (0)

template <typename T> static T private_sym(const char* name)
{
    void* p = dlsym(RTLD_DEFAULT, name);
    if (!p) {
        fprintf(stderr, T_TAG ": private symbol %s not found\n", name);
        exit(1);
    }
    return reinterpret_cast<T>(p);
}

static bool same_channels(MTLTextureSwizzleChannels a,
                          MTLTextureSwizzleChannels b)
{
    return a.red == b.red && a.green == b.green && a.blue == b.blue &&
           a.alpha == b.alpha;
}

static void test_swizzle()
{
    auto c2k = private_sym<uint32_t (*)(MTLTextureSwizzleChannels)>(
        "MTLTextureSwizzleChannelsToKey");
    auto k2c = private_sym<MTLTextureSwizzleChannels (*)(uint32_t)>(
        "MTLTextureSwizzleKeyToChannels");
    int n = 0;
    for (int r = 0; r < 6; r++)
        for (int g = 0; g < 6; g++)
            for (int b = 0; b < 6; b++)
                for (int a = 0; a < 6; a++) {
                    MTLTextureSwizzleChannels c = MTLTextureSwizzleChannelsMake(
                        (MTLTextureSwizzle)r, (MTLTextureSwizzle)g,
                        (MTLTextureSwizzle)b, (MTLTextureSwizzle)a);
                    uint32_t key = c2k(c);
                    EXPECT(dmn_swizzle_channels_to_key(c) == key,
                           "ChannelsToKey(%d,%d,%d,%d)", r, g, b, a);
                    EXPECT(same_channels(dmn_swizzle_key_to_channels(key),
                                         k2c(key)),
                           "KeyToChannels(0x%08x)", key);
                    n++;
                }
    for (uint32_t key : {0u, 0xffffffffu, 0xffc80907u, 0x01020304u})
        EXPECT(same_channels(dmn_swizzle_key_to_channels(key), k2c(key)),
               "KeyToChannels(0x%08x)", key);
    printf(T_TAG ": swizzle: %d combinations\n", n);
}

static void compare_image(const char* what, const void* addr)
{
    auto hdr = private_sym<const struct mach_header* (*)(const void*)>(
        "dyld_image_header_containing_address");
    auto uuid = private_sym<bool (*)(const struct mach_header*, uuid_t)>(
        "_dyld_get_image_uuid");

    const struct mach_header* mh = hdr(addr);
    EXPECT(mh, "%s: dyld found no image", what);
    EXPECT(dmn_image_header_containing(addr) == mh, "%s: header %p != %p",
           what, (const void*)dmn_image_header_containing(addr),
           (const void*)mh);
    if (!mh)
        return;

    uuid_t want, got;
    uuid_clear(want);
    uuid_clear(got);
    bool want_ok = uuid(mh, want);
    bool got_ok = dmn_image_uuid(mh, got);
    EXPECT(want_ok && got_ok, "%s: uuid lookup failed", what);
    EXPECT(!uuid_compare(want, got), "%s: uuid mismatch", what);
    uuid_string_t s;
    uuid_unparse(got, s);
    printf(T_TAG ": image %s: %s\n", what, s);
}

static void test_images(const std::string& fw)
{
    void* d3dm = dlopen(fw.c_str(), RTLD_NOW | RTLD_NOLOAD);
    EXPECT(d3dm, "D3DMetal is not loaded");
    if (d3dm) {
        void* fn = dlsym(d3dm, "D3D12CreateDevice");
        EXPECT(fn, "D3DMetal has no D3D12CreateDevice");
        if (fn) {
            compare_image("D3DMetal", fn);
            compare_image("D3DMetal+64", (const char*)fn + 64);
        }
    }

    std::string dir = fw.substr(0, fw.rfind('/'));
    std::string irc = dir + "/Resources/libmetalirconverter.dylib";
    void* ir = dlopen(irc.c_str(), RTLD_NOW | RTLD_NOLOAD);
    EXPECT(ir, "libmetalirconverter is not loaded (%s)", irc.c_str());
    if (ir) {
        void* fn = dlsym(ir, "_Z25IRCompilerSetTargetTripleP10IRCompilerPKc");
        EXPECT(fn, "libmetalirconverter has no IRCompilerSetTargetTriple");
        if (fn)
            compare_image("libmetalirconverter", fn);
    }

    compare_image("executable", (const void*)&test_images);

    void* heap = malloc(64);
    EXPECT(!dmn_image_header_containing(heap),
           "heap address resolved to an image");
    free(heap);
    EXPECT(!dmn_image_header_containing(nullptr), "NULL resolved to an image");
    uuid_t u;
    EXPECT(!dmn_image_uuid(nullptr, u), "NULL header has a uuid");
}

static id<MTLAccelerationStructure> alloc_as(id<MTLDevice> dev, id desc)
{
    ((void (*)(id, SEL, MTLStorageMode))objc_msgSend)(
        desc, @selector(setStorageMode:), MTLStorageModePrivate);
    return ((id(*)(id, SEL, NSUInteger, id))objc_msgSend)(
        dev, @selector(newAccelerationStructureWithSize:withDescriptor:),
        (NSUInteger)65536, desc);
}

static int g_set_storage = 0;

static void test_as_descriptor()
{
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev.supportsRaytracing) {
            printf(T_TAG ": as: device has no ray tracing, skipped\n");
            return;
        }
        Class real = NSClassFromString(
            @"MTLAccelerationStructureAllocationDescriptor");
        Class ours = NSClassFromString(
            @"DMNAccelerationStructureAllocationDescriptor");
        EXPECT(real && ours, "descriptor classes missing");
        if (!real || !ours)
            return;
        EXPECT(!strcmp(class_getName(class_getSuperclass(ours)),
                       class_getName(class_getSuperclass(real))),
               "superclass differs");
        for (const char* sel :
             {"storageMode", "setStorageMode:", "forceResourceIndex",
              "setForceResourceIndex:", "resourceIndex", "setResourceIndex:",
              "copyWithZone:", "isEqual:", "hash"})
            EXPECT(class_getInstanceMethod(ours, sel_registerName(sel)),
                   "descriptor lacks -%s", sel);

        id rd = [[real alloc] init];
        id od = [[ours alloc] init];
        EXPECT([od storageMode] == [rd storageMode], "default storageMode");
        id<MTLAccelerationStructure> ra = alloc_as(dev, rd);
        id<MTLAccelerationStructure> oa = alloc_as(dev, od);
        EXPECT(ra && oa, "allocation failed (real %p, ours %p)", ra, oa);
        if (ra && oa) {
            EXPECT(ra.size == oa.size, "size %lu != %lu",
                   (unsigned long)oa.size, (unsigned long)ra.size);
            EXPECT(oa.storageMode == MTLStorageModePrivate,
                   "storage mode not honoured");
            EXPECT(oa.gpuResourceID._impl != 0, "no resource id");
        }
        id copy = [od copy];
        EXPECT([copy isEqual:od] && [copy hash] == [od hash], "copy differs");
        [copy release];
        [ra release];
        [oa release];
        [rd release];
        [od release];

        /* Count D3DMetal's own use of the class through a D3D12 create. */
        Method m = class_getInstanceMethod(ours, @selector(setStorageMode:));
        auto orig = (void (*)(id, SEL, MTLStorageMode))method_getImplementation(m);
        method_setImplementation(
            m, imp_implementationWithBlock(^(id self_, MTLStorageMode mode) {
                g_set_storage++;
                orig(self_, @selector(setStorageMode:), mode);
            }));
        unsigned long long va = 0;
        int rc = import_shims_create_as(&va);
        method_setImplementation(m, (IMP)orig);
        if (rc < 0) {
            printf(T_TAG ": as: D3D12 reports no ray tracing, skipped\n");
            return;
        }
        EXPECT(rc == 0 && va, "D3D12 acceleration-structure create rc=%d", rc);
        /* D3DMetal 3.0 only takes the descriptor path for CPU-accessible
         * acceleration structures, which it never enables; a patched
         * framework that loaded at all has bound the class. */
        printf(T_TAG ": as: D3D12 create va=0x%llx, our class used %d time(s)\n",
               va, g_set_storage);
    }
}

int main()
{
    @autoreleasepool {
        test_swizzle();

        /* The no-op shims must be callable and must not run the payload. */
        __block bool ran = false;
        dmn_analytics_send(@"com.example.test", ^NSDictionary*(void) {
            ran = true;
            return @{};
        });
        EXPECT(!ran, "analytics payload block ran");

        if (dmn_init(nullptr) != DMN_SUCCESS) {
            fprintf(stderr, T_TAG ": dmn_init failed\n");
            return 1;
        }
        std::string fw = dmn_framework_path();
        NSData* bin = [NSData dataWithContentsOfFile:@(fw.c_str())];
        bool patched =
            [bin rangeOfData:[NSData dataWithBytes:"_dmn_image_uuid" length:16]
                     options:0
                       range:NSMakeRange(0, bin.length)]
                .location != NSNotFound;
        printf(T_TAG ": %s D3DMetal: %s\n", patched ? "patched" : "unpatched",
               fw.c_str());

        test_images(fw);
        test_as_descriptor();
    }
    if (g_fail) {
        fprintf(stderr, T_TAG ": %d failure(s)\n", g_fail);
        return 1;
    }
    printf(T_TAG ": PASS\n");
    return 0;
}
