#ifndef PHONON2_RUNTIME_H
#define PHONON2_RUNTIME_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

#define PHONON_EXPORT __attribute__((visibility("default")))

struct Phonon2Context {
    int out_features;
    int in_features;
    int num_threads;
    uint8_t* sign_codes;
    bool* is_hi;
    float* lo;
    float* hi;
    int mode;
};

PHONON_EXPORT int phonon2_cpu_abi_version();
PHONON_EXPORT int phonon2_cpu_kernel_in_use();
PHONON_EXPORT void phonon2_cpu_set_kernel(int k);
PHONON_EXPORT void phonon2_cpu_set_spin_iters(int iters);
PHONON_EXPORT int phonon2_cpu_set_threads(int threads);
PHONON_EXPORT float phonon2_cpu_enable_onedot(void* ctx);
PHONON_EXPORT void phonon2_cpu_set_onedot(void* ctx, int on);

PHONON_EXPORT void* phonon2_cpu_create(int out_features, int in_features,
                                       void* sign_codes, void* is_hi,
                                       void* lo_scales, void* hi_scales,
                                       int mode);
PHONON_EXPORT void phonon2_cpu_destroy(void* handle);
PHONON_EXPORT void phonon2_cpu_dequant_row(void* handle, int row, float* dst);
PHONON_EXPORT int phonon2_cpu_gemm(void* handle, const float* input, int m, float* output, int ldc);

#ifdef __cplusplus
}
#endif

#endif // PHONON2_RUNTIME_H
