/*
 * DXIL container normalization (see dmn_dxil_reflect.cpp).
 */

#ifndef DMN_DXIL_REFLECT_H
#define DMN_DXIL_REFLECT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Normalize supported DXIL containers. Subroutine inlining covers D3D12
 * in-driver DXBC stage kinds 0 through 5 and the guest's original
 * one-DXIL-part, zero-digest container shape. QuadVote lowering is
 * module-generic for supported stages and leaves shader-model and DXIL
 * version fields unchanged. Returns 1 with a malloc-owned container in *out
 * / *out_size only when bytes change; returns 0 with *out == NULL when the
 * input is kept unchanged. */
int dmn_dxil_normalize(const void *container, size_t size, void **out,
                       size_t *out_size);

#ifdef __cplusplus
}
#endif

#endif /* DMN_DXIL_REFLECT_H */
