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

/* For a minimal container (one DXIL part, zero digest -- the shape the guest
 * UMD wraps a bare DXIL program in) whose module defines internal
 * functions, inline them and turn the globals they shared into locals.
 * Returns 1 and a malloc'd container in *out / *out_size (caller frees)
 * when it rewrote the module, 0 to keep the input. */
int dmn_dxil_inline_subroutines(const void *container, size_t size,
                                void **out, size_t *out_size);

#ifdef __cplusplus
}
#endif

#endif /* DMN_DXIL_REFLECT_H */
