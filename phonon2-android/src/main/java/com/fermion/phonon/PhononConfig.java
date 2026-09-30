package com.fermion.phonon;

public class PhononConfig {
    private int numThreads = 4;
    private boolean preferNeon = true;

    public PhononConfig() {}

    public PhononConfig(int numThreads) {
        this.numThreads = numThreads;
    }

    public int getNumThreads() {
        return numThreads;
    }

    public void setNumThreads(int numThreads) {
        this.numThreads = numThreads;
    }

    public boolean isPreferNeon() {
        return preferNeon;
    }

    public void setPreferNeon(boolean preferNeon) {
        this.preferNeon = preferNeon;
    }
}
