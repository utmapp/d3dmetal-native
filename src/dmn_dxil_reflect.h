/*
 * DXIL container reflection and normalization (see dmn_dxil_reflect.cpp).
 */

#ifndef DMN_DXIL_REFLECT_H
#define DMN_DXIL_REFLECT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct dmn_dxil_semantic {
    uint32_t reg;    /* input-signature register (row) */
    uint32_t index;  /* SemanticIndex */
    char name[64];   /* SemanticName */
};

/* Fill out[] with the input-signature semantics of the DXIL container.
 * Returns the number of entries written (0 on any failure). */
int dmn_dxil_input_semantics(const void *container, size_t size,
                             struct dmn_dxil_semantic *out, int cap);

/* Normalize supported DXIL containers. Internal-function inlining remains
 * limited to the guest's original one-DXIL-part, zero-digest container shape;
 * QuadVote lowering is module-generic for supported stages and is applied
 * without changing shader-model or DXIL-version fields. Returns 1 with a
 * malloc-owned container in *out / *out_size only when bytes changed; returns
 * 0 with *out == NULL when the input is kept unchanged. */
int dmn_dxil_normalize(const void *container, size_t size, void **out,
                       size_t *out_size);

#ifdef __cplusplus
}
#endif

#endif /* DMN_DXIL_REFLECT_H */
