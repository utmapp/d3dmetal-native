/*
 * Copyright 2026 Turing Software LLC
 * SPDX-License-Identifier: MIT
 *
 * See callback_embedder.h. Manual reference counting (-fno-objc-arc).
 */

#import <Metal/Metal.h>

#include <cstring>

#include "callback_embedder.h"

namespace {

constexpr uint32_t kSlots = 3;

struct Embedder {
    id<MTLDevice> device = nil;
    id<MTLTexture> tex[kSlots] = {};
    uint32_t cfg_w = 0, cfg_h = 0, cfg_calls = 0;
    uint32_t next_slot = 0;
    uint32_t acquires = 0, presents = 0;
    uint32_t last_presented = UINT32_MAX;
};

bool emb_configure(void* ctx, uint32_t w, uint32_t h, uint32_t fmt,
                   uint32_t buffers) {
    (void)fmt; (void)buffers;
    auto* e = static_cast<Embedder*>(ctx);
    e->cfg_w = w;
    e->cfg_h = h;
    e->cfg_calls++;
    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                     width:w
                                    height:h
                                 mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
               MTLTextureUsageShaderWrite;
    td.storageMode = MTLStorageModeShared; /* CPU-verifiable */
    for (uint32_t i = 0; i < kSlots; i++) {
        [e->tex[i] release];
        e->tex[i] = [e->device newTextureWithDescriptor:td];
        if (!e->tex[i])
            return false;
    }
    return true;
}

void* emb_acquire(void* ctx, uint32_t* out_slot) {
    auto* e = static_cast<Embedder*>(ctx);
    uint32_t s = e->next_slot;
    e->next_slot = (e->next_slot + 1) % kSlots;
    e->acquires++;
    *out_slot = s;
    return (void*)e->tex[s]; /* borrowed */
}

void emb_present(void* ctx, uint32_t slot) {
    auto* e = static_cast<Embedder*>(ctx);
    e->last_presented = slot;
    e->presents++;
}

} // namespace

void* cbemb_create(void) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev)
        return nullptr;
    auto* e = new Embedder();
    e->device = dev; /* +1 from create */
    return e;
}

void cbemb_destroy(void* emb) {
    auto* e = static_cast<Embedder*>(emb);
    if (!e)
        return;
    for (uint32_t i = 0; i < kSlots; i++)
        [e->tex[i] release];
    [e->device release];
    delete e;
}

void cbemb_fill_callbacks(void* emb, dmn_window_callbacks* out) {
    memset(out, 0, sizeof(*out));
    out->struct_size = sizeof(*out);
    out->ctx = emb;
    out->configure = emb_configure;
    out->acquire_texture = emb_acquire;
    out->present = emb_present;
}

uint32_t cbemb_configures(void* emb) { return static_cast<Embedder*>(emb)->cfg_calls; }
uint32_t cbemb_config_w(void* emb)   { return static_cast<Embedder*>(emb)->cfg_w; }
uint32_t cbemb_config_h(void* emb)   { return static_cast<Embedder*>(emb)->cfg_h; }
uint32_t cbemb_acquires(void* emb)   { return static_cast<Embedder*>(emb)->acquires; }
uint32_t cbemb_presents(void* emb)   { return static_cast<Embedder*>(emb)->presents; }
uint32_t cbemb_last_slot(void* emb)  { return static_cast<Embedder*>(emb)->last_presented; }

bool cbemb_read_center(void* emb, uint8_t out[4]) {
    auto* e = static_cast<Embedder*>(emb);
    if (e->last_presented >= kSlots || !e->tex[e->last_presented])
        return false;
    id<MTLTexture> t = e->tex[e->last_presented];
    [t getBytes:out
      bytesPerRow:4
       fromRegion:MTLRegionMake2D(e->cfg_w / 2, e->cfg_h / 2, 1, 1)
      mipmapLevel:0];
    return true;
}
