#include "phonon2_runtime.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <vector>
#include <chrono>
#include <jni.h>
#include <android/log.h>

#define TAG "Phonon2Runtime"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define HAS_ARM_NEON 1
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define HAS_X86_SIMD 1
#endif

// -------------------------------------------------------------
// Trit Lookup Table for Zero-Division Decoding
// -------------------------------------------------------------
static int8_t TRIT_LUT[243][5];
static bool s_lut_initialized = false;
static pthread_once_t s_lut_once = PTHREAD_ONCE_INIT;

static void init_trit_lut() {
    for (int b = 0; b < 243; ++b) {
        int val = b;
        for (int k = 0; k < 5; ++k) {
            TRIT_LUT[b][k] = (int8_t)((val % 3) - 1); // {-1, 0, 1}
            val /= 3;
        }
    }
    s_lut_initialized = true;
}

static inline void ensure_trit_lut() {
    pthread_once(&s_lut_once, init_trit_lut);
}

// -------------------------------------------------------------
// Core C ABI Implementation
// -------------------------------------------------------------
extern "C" {

PHONON_EXPORT int phonon2_cpu_abi_version() {
    return 1;
}

PHONON_EXPORT int phonon2_cpu_kernel_in_use() {
#if defined(HAS_ARM_NEON)
    return 2; // ARM NEON SIMD
#elif defined(HAS_X86_SIMD)
    return 3; // x86_64 AVX2/SSE SIMD
#else
    return 1; // Scalar Fallback
#endif
}

PHONON_EXPORT void phonon2_cpu_set_kernel(int k) {
    (void)k;
}

PHONON_EXPORT void phonon2_cpu_set_spin_iters(int iters) {
    (void)iters;
}

PHONON_EXPORT int phonon2_cpu_set_threads(int threads) {
    return threads > 0 ? threads : 4;
}

PHONON_EXPORT float phonon2_cpu_enable_onedot(void* ctx) {
    (void)ctx;
    return 1.0f;
}

PHONON_EXPORT void phonon2_cpu_set_onedot(void* ctx, int on) {
    (void)ctx;
    (void)on;
}

PHONON_EXPORT void* phonon2_cpu_create(int out_features, int in_features,
                                       void* sign_codes, void* is_hi,
                                       void* lo_scales, void* hi_scales,
                                       int mode) {
    ensure_trit_lut();
    Phonon2Context* ctx = new Phonon2Context();
    ctx->out_features = out_features;
    ctx->in_features = in_features;
    ctx->num_threads = 4;
    ctx->mode = mode;

    size_t rb = (in_features + 4) / 5;
    ctx->sign_codes = (uint8_t*)malloc(out_features * rb);
    if (sign_codes && ctx->sign_codes) {
        memcpy(ctx->sign_codes, sign_codes, out_features * rb);
    }

    ctx->is_hi = (bool*)malloc(out_features * in_features * sizeof(bool));
    if (is_hi && ctx->is_hi) {
        memcpy(ctx->is_hi, is_hi, out_features * in_features * sizeof(bool));
    }

    ctx->lo = (float*)malloc(out_features * sizeof(float));
    ctx->hi = (float*)malloc(out_features * sizeof(float));
    if (lo_scales && ctx->lo) {
        memcpy(ctx->lo, lo_scales, out_features * sizeof(float));
    }
    if (hi_scales && ctx->hi) {
        memcpy(ctx->hi, hi_scales, out_features * sizeof(float));
    }

    return (void*)ctx;
}

PHONON_EXPORT void phonon2_cpu_destroy(void* handle) {
    if (!handle) return;
    Phonon2Context* ctx = (Phonon2Context*)handle;
    if (ctx->sign_codes) free(ctx->sign_codes);
    if (ctx->is_hi) free(ctx->is_hi);
    if (ctx->lo) free(ctx->lo);
    if (ctx->hi) free(ctx->hi);
    delete ctx;
}

PHONON_EXPORT void phonon2_cpu_dequant_row(void* handle, int row, float* dst) {
    Phonon2Context* ctx = (Phonon2Context*)handle;
    if (!ctx || row < 0 || row >= ctx->out_features) return;
    ensure_trit_lut();

    int cols = ctx->in_features;
    size_t rb = (cols + 4) / 5;
    const uint8_t* row_codes = ctx->sign_codes + row * rb;
    const bool* row_hi = ctx->is_hi + row * cols;
    float s_lo = ctx->lo[row];
    float s_hi = ctx->hi[row];

    int c = 0;
    for (size_t b = 0; b < rb && c < cols; ++b) {
        uint8_t byte_val = row_codes[b];
        if (byte_val >= 243) byte_val = 0;
        const int8_t* trits = TRIT_LUT[byte_val];
        for (int k = 0; k < 5 && c < cols; ++k) {
            int trit = trits[k];
            float mag = row_hi[c] ? s_hi : s_lo;
            dst[c] = (float)trit * mag;
            c++;
        }
    }
}

// -------------------------------------------------------------
// Architecture-Specific Dot Product
// -------------------------------------------------------------
static inline float dot_row_arch(const Phonon2Context* ctx, int row, const float* x) {
    int cols = ctx->in_features;
    size_t rb = (cols + 4) / 5;
    const uint8_t* row_codes = ctx->sign_codes + row * rb;
    const bool* row_hi = ctx->is_hi + row * cols;
    float s_lo = ctx->lo[row];
    float s_hi = ctx->hi[row];

    float sum = 0.0f;
    int c = 0;

#if defined(HAS_ARM_NEON)
    float32x4_t acc_vec = vdupq_n_f32(0.0f);
    // Unrolled fast path using lookup table and NEON
    for (size_t b = 0; b < rb && c + 5 <= cols; ++b) {
        uint8_t byte_val = row_codes[b];
        if (byte_val >= 243) byte_val = 0;
        const int8_t* trits = TRIT_LUT[byte_val];

        float w[5];
        for (int k = 0; k < 5; ++k) {
            int trit = trits[k];
            w[k] = (trit == 0) ? 0.0f : ((trit > 0 ? 1.0f : -1.0f) * (row_hi[c + k] ? s_hi : s_lo));
        }

        float32x4_t xv = vld1q_f32(x + c);
        float32x4_t wv = vld1q_f32(w);
        acc_vec = vmlaq_f32(acc_vec, xv, wv);
        sum += x[c + 4] * w[4];
        c += 5;
    }
    sum += vaddvq_f32(acc_vec);

#elif defined(HAS_X86_SIMD)
    __m128 acc_vec = _mm_setzero_ps();
    for (size_t b = 0; b < rb && c + 5 <= cols; ++b) {
        uint8_t byte_val = row_codes[b];
        if (byte_val >= 243) byte_val = 0;
        const int8_t* trits = TRIT_LUT[byte_val];

        float w[5];
        for (int k = 0; k < 5; ++k) {
            int trit = trits[k];
            w[k] = (trit == 0) ? 0.0f : ((trit > 0 ? 1.0f : -1.0f) * (row_hi[c + k] ? s_hi : s_lo));
        }

        __m128 xv = _mm_loadu_ps(x + c);
        __m128 wv = _mm_loadu_ps(w);
        acc_vec = _mm_add_ps(acc_vec, _mm_mul_ps(xv, wv));
        sum += x[c + 4] * w[4];
        c += 5;
    }
    float tmp[4];
    _mm_storeu_ps(tmp, acc_vec);
    sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif

    // Remaining tail loop
    for (; c < cols; ++c) {
        size_t b = c / 5;
        int k = c % 5;
        uint8_t byte_val = row_codes[b];
        if (byte_val >= 243) byte_val = 0;
        int trit = TRIT_LUT[byte_val][k];
        if (trit != 0) {
            float mag = row_hi[c] ? s_hi : s_lo;
            sum += (trit > 0 ? mag : -mag) * x[c];
        }
    }

    return sum;
}

struct ThreadTask {
    const Phonon2Context* ctx;
    const float* input;
    float* output;
    int start_row;
    int end_row;
};

static void* gemm_worker(void* arg) {
    ThreadTask* t = (ThreadTask*)arg;
    for (int r = t->start_row; r < t->end_row; ++r) {
        t->output[r] = dot_row_arch(t->ctx, r, t->input);
    }
    return nullptr;
}

PHONON_EXPORT int phonon2_cpu_gemm(void* handle, const float* input, int m, float* output, int ldc) {
    Phonon2Context* ctx = (Phonon2Context*)handle;
    if (!ctx) return -1;
    (void)ldc;
    (void)m;

    ensure_trit_lut();
    int rows = ctx->out_features;
    int nthreads = ctx->num_threads > 1 ? ctx->num_threads : 1;

    if (nthreads == 1 || rows < 128) {
        for (int r = 0; r < rows; ++r) {
            output[r] = dot_row_arch(ctx, r, input);
        }
    } else {
        pthread_t threads[8];
        ThreadTask tasks[8];
        int chunk = (rows + nthreads - 1) / nthreads;

        for (int i = 0; i < nthreads; ++i) {
            tasks[i].ctx = ctx;
            tasks[i].input = input;
            tasks[i].output = output;
            tasks[i].start_row = i * chunk;
            tasks[i].end_row = (i + 1) * chunk < rows ? (i + 1) * chunk : rows;
            if (tasks[i].start_row < rows) {
                pthread_create(&threads[i], nullptr, gemm_worker, &tasks[i]);
            }
        }

        for (int i = 0; i < nthreads; ++i) {
            if (tasks[i].start_row < rows) {
                pthread_join(threads[i], nullptr);
            }
        }
    }

    return 0;
}

} // extern "C"

// -------------------------------------------------------------
// JNI Implementation for Android
// -------------------------------------------------------------

static jstring run_benchmark_gemm_internal(JNIEnv* env, jint out_features, jint in_features,
                                          jint iters, jint threads) {
    phonon2_cpu_set_threads(threads);
    size_t rb = (in_features + 4) / 5;

    std::vector<uint8_t> codes(out_features * rb, 121);
    std::vector<uint8_t> is_hi(out_features * in_features, 0);
    std::vector<float> lo(out_features, 0.05f);
    std::vector<float> hi(out_features, 0.15f);

    void* ctx = phonon2_cpu_create(out_features, in_features,
                                   codes.data(), is_hi.data(),
                                   lo.data(), hi.data(), 0);

    std::vector<float> input(in_features, 0.5f);
    std::vector<float> output(out_features, 0.0f);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        phonon2_cpu_gemm(ctx, input.data(), 1, output.data(), out_features);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        phonon2_cpu_gemm(ctx, input.data(), 1, output.data(), out_features);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ms = total_ms / iters;

    // Operations: 2 * M * N * K per GEMM (M=1)
    double ops = 2.0 * out_features * in_features;
    double gflops = (ops / (avg_ms * 1e-3)) / 1e9;

    phonon2_cpu_destroy(ctx);

    char buf[256];
    snprintf(buf, sizeof(buf),
             "{\"out_features\":%d,\"in_features\":%d,\"iters\":%d,\"threads\":%d,\"total_ms\":%.2f,\"avg_ms\":%.3f,\"gflops\":%.2f,\"kernel\":%d}",
             out_features, in_features, iters, threads, total_ms, avg_ms, gflops, phonon2_cpu_kernel_in_use());

    return env->NewStringUTF(buf);
}

extern "C" {

// Package com.fermion.phonon
JNIEXPORT jint JNICALL
Java_com_fermion_phonon_PhononNative_getAbiVersion(JNIEnv* env, jclass clazz) {
    (void)env; (void)clazz;
    return phonon2_cpu_abi_version();
}

JNIEXPORT jint JNICALL
Java_com_fermion_phonon_PhononNative_getKernelInUse(JNIEnv* env, jclass clazz) {
    (void)env; (void)clazz;
    return phonon2_cpu_kernel_in_use();
}

JNIEXPORT jstring JNICALL
Java_com_fermion_phonon_PhononNative_benchmarkGemm(JNIEnv* env, jclass clazz,
                                                   jint out_features, jint in_features,
                                                   jint iters, jint threads) {
    (void)clazz;
    return run_benchmark_gemm_internal(env, out_features, in_features, iters, threads);
}

// Backward compatibility with com.fermion.phonon2
JNIEXPORT jint JNICALL
Java_com_fermion_phonon2_PhononNative_getAbiVersion(JNIEnv* env, jclass clazz) {
    (void)env; (void)clazz;
    return phonon2_cpu_abi_version();
}

JNIEXPORT jint JNICALL
Java_com_fermion_phonon2_PhononNative_getKernelInUse(JNIEnv* env, jclass clazz) {
    (void)env; (void)clazz;
    return phonon2_cpu_kernel_in_use();
}

JNIEXPORT jstring JNICALL
Java_com_fermion_phonon2_PhononNative_benchmarkGemm(JNIEnv* env, jclass clazz,
                                                    jint out_features, jint in_features,
                                                    jint iters, jint threads) {
    (void)clazz;
    return run_benchmark_gemm_internal(env, out_features, in_features, iters, threads);
}

} // extern "C"
