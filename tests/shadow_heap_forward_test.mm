/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * shadow_heap_forward_test: DmnShadowHeap must forward selectors it does not
 * implement to the buffer it stands in for.
 *
 * The impostor is duck-typed -- it implements only the handful of selectors
 * D3DMetal actually sends a heap. Anything else that talks to Metal objects
 * sends more. With MTL_DEBUG_LAYER=1, -[MTLDebugCommandBuffer preCommit] walks
 * every resource in the command buffer and sends each -lockPurgeableState,
 * which raised
 *     -[DmnShadowHeap lockPurgeableState]: unrecognized selector sent to ...
 * and took the render-server worker down with SIGABRT -- so the debug layer,
 * and Metal shader validation with it, could not be used on exactly the GPU
 * page-fault bugs they exist to diagnose.
 *
 * This drives the contract directly through the ObjC runtime rather than by
 * standing up a device and a shared texture: the failure is a property of the
 * class, and reaching it this way keeps the test hermetic (no D3DMetal
 * framework, no GPU submission, no MTL_DEBUG_LAYER-version dependence).
 *
 * Fails before the fix (unrecognized selector) and passes after it.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <objc/runtime.h>
#include <objc/message.h>

#include <cstdio>

#include "d3dmetal_native.h"

#define T_TAG "SHADOWFWD"
#include "common/check.h"

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            fprintf(stderr, T_TAG ": no Metal device; SKIP\n");
            return 77;              /* meson: skipped */
        }

        /* Touch one library symbol so the linker keeps the dependency on
         * libd3dmetal-native: everything below reaches the class through the
         * ObjC runtime by NAME, which references no symbol, and the dylib was
         * dead-stripped out of the link without this. It also states the
         * hermeticity the test relies on -- the library is never initialised,
         * so no D3DMetal framework has to be present. */
        EXPECT(!dmn_is_initialized(), "library unexpectedly already initialised");

        /* The class lives inside libd3dmetal-native; the runtime registers it
         * by name when the dylib loads, so no library entry point is needed. */
        Class cls = objc_getClass("DmnShadowHeap");
        EXPECT(cls != nullptr, "DmnShadowHeap class is not registered");

        Ivar give = class_getInstanceVariable(cls, "_give");
        Ivar claim = class_getInstanceVariable(cls, "_sizeClaim");
        EXPECT(give != nullptr && claim != nullptr,
               "DmnShadowHeap no longer exposes _give/_sizeClaim");

        const NSUInteger kLen = 4096;
        id<MTLBuffer> backing =
            [dev newBufferWithLength:kLen options:MTLResourceStorageModeShared];
        EXPECT(backing != nil, "could not allocate the backing buffer");

        id sh = [[cls alloc] init];
        EXPECT(sh != nullptr, "could not allocate the impostor");
        object_setIvar(sh, give, backing);
        *(NSUInteger *)((char *)sh + ivar_getOffset(claim)) = kLen;

        /* -size IS implemented by the impostor: the baseline. */
        const NSUInteger size =
            ((NSUInteger (*)(id, SEL))objc_msgSend)(sh, @selector(size));
        EXPECT(size == kLen, "-size (implemented) did not answer the claim");

        /* -length is NOT implemented by the impostor, but MTLBuffer has it.
         * Before the fix this raises NSInvalidArgumentException. Catching it
         * turns the abort into a legible failure rather than SIGABRT. */
        NSUInteger len = 0;
        @try {
            len = ((NSUInteger (*)(id, SEL))objc_msgSend)(sh, @selector(length));
        } @catch (NSException *e) {
            fprintf(stderr, T_TAG ": FAIL unimplemented selector raised %s: %s\n",
                    [[e name] UTF8String], [[e reason] UTF8String]);
            return 1;
        }
        EXPECT(len == kLen,
               "-length was not forwarded to the backing buffer");

        /* The fix must NOT start advertising these selectors: D3DMetal probes
         * capabilities with -respondsToSelector:, and answering YES would
         * change which paths it takes. Forwarding is for calls that would
         * otherwise throw, nothing more. */
        EXPECT(![(id)sh respondsToSelector:@selector(length)],
               "-respondsToSelector: now answers YES; D3DMetal capability "
               "probes would change behaviour");

        /* A selector nothing implements must still raise rather than silently
         * return garbage -- forwarding is a fallback, not a black hole. */
        bool raised = false;
        @try {
            ((void (*)(id, SEL))objc_msgSend)(sh, @selector(dmnNoSuchSelector));
        } @catch (NSException *e) {
            raised = true;
        }
        EXPECT(raised,
               "a selector nobody implements was swallowed instead of raising");

        [backing release];
        T_PASS();
        return 0;
    }
}
