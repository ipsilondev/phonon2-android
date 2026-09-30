package com.fermion.phonon2;

/**
 * Backward compatibility alias for PhononNative.
 */
public final class PhononNative {

    private static boolean isLoaded = false;
    private static String loadError = null;

    static {
        try {
            System.loadLibrary("phonon2_android");
            isLoaded = true;
        } catch (Throwable t) {
            isLoaded = false;
            loadError = t.getMessage();
        }
    }

    public static boolean isAvailable() {
        return isLoaded;
    }

    public static String getLoadError() {
        return loadError;
    }

    public static native int getAbiVersion();

    public static native int getKernelInUse();

    public static native String benchmarkGemm(int outFeatures, int inFeatures, int iters, int threads);

    private PhononNative() {}
}
