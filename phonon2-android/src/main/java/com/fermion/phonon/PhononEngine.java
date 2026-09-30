package com.fermion.phonon;

/**
 * High-level engine interface for Fermion Phonon-2 on Android.
 */
public class PhononEngine {

    private final PhononConfig config;

    public PhononEngine() {
        this(new PhononConfig());
    }

    public PhononEngine(PhononConfig config) {
        this.config = config != null ? config : new PhononConfig();
    }

    public boolean isReady() {
        return PhononNative.isAvailable();
    }

    public String getKernelName() {
        if (!isReady()) return "Unavailable";
        int kernel = PhononNative.getKernelInUse();
        switch (kernel) {
            case 2:
                return "ARM NEON SIMD (Vectorized)";
            case 3:
                return "x86_64 AVX2/SSE SIMD (Vectorized)";
            default:
                return "Scalar Fallback";
        }
    }

    public int getAbiVersion() {
        return isReady() ? PhononNative.getAbiVersion() : -1;
    }

    public String runBenchmark(int outFeatures, int inFeatures, int iters) {
        if (!isReady()) {
            throw new IllegalStateException("Phonon native runtime is not loaded: " + PhononNative.getLoadError());
        }
        return PhononNative.benchmarkGemm(outFeatures, inFeatures, iters, config.getNumThreads());
    }
}
