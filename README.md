# Phonon-2 Android Library (`phonon2-android`)

Android AAR library bringing **Fermion Phonon-2** 2.1-bit ternary ASR (FastConformer + TDT) native inference to Android devices and emulators.

## Supported Architectures

| Architecture | SIMD Kernel | Status | Target Hardware |
| :--- | :--- | :--- | :--- |
| **`arm64-v8a`** | **ARM NEON SIMD** | Optimized | Physical 64-bit ARM devices (Cortex-A, Kryo, Tensor) |
| **`x86_64`** | **AVX2 / SSE4.2 SIMD** | Optimized | Android Studio Emulators & x86_64 Chromebooks |

Native binary size: **~350 KB** per architecture (zero external dependencies, links statically with Bionic libc++).

---

## Building the AAR

Prerequisites:
- Android SDK (API 34)
- Android NDK (r26+ or r27+)
- CMake 3.22.1+

Run Gradle to build the release archive:

```bash
# On Windows
./gradlew.bat assembleRelease

# On Linux / macOS
./gradlew assembleRelease
```

The output `.aar` artifact is produced at:
```text
phonon2-android/build/outputs/aar/phonon2-android-release.aar
```

Inside the `.aar`:
```text
jni/
├── arm64-v8a/
│   └── libphonon2_android.so   (ARM NEON SIMD)
└── x86_64/
    └── libphonon2_android.so   (AVX2 / SSE SIMD)
classes.jar
└── com/fermion/
    ├── phonon/
    │   ├── PhononConfig.class
    │   ├── PhononEngine.class
    │   └── PhononNative.class
    └── phonon2/
        └── PhononNative.class
```

---

## Installation & Integration

### Option 1: Direct AAR Dependency

Copy `phonon2-android-release.aar` into your app's `libs/` directory, and add to `app/build.gradle`:

```groovy
dependencies {
    implementation files('libs/phonon2-android-release.aar')
}
```

### Option 2: Project Dependency

Include in `settings.gradle`:
```groovy
include ':phonon2-android'
project(':phonon2-android').projectDir = new File('/path/to/phonon2-android/phonon2-android')
```

And in `app/build.gradle`:
```groovy
dependencies {
    implementation project(':phonon2-android')
}
```

---

## API Usage

### High-Level API (`PhononEngine`)

```java
import com.fermion.phonon.PhononEngine;
import com.fermion.phonon.PhononConfig;

PhononConfig config = new PhononConfig(4); // 4 worker threads
PhononEngine engine = new PhononEngine(config);

if (engine.isReady()) {
    Log.i("Phonon", "Kernel: " + engine.getKernelName());
    Log.i("Phonon", "ABI Version: " + engine.getAbiVersion());

    // Run timed 5-value GEMM benchmark (out=4096, in=1024, iters=200)
    String reportJson = engine.runBenchmark(4096, 1024, 200);
    Log.i("Phonon", "Benchmark: " + reportJson);
}
```

### Low-Level JNI API (`PhononNative`)

```java
import com.fermion.phonon.PhononNative;

if (PhononNative.isAvailable()) {
    int abi = PhononNative.getAbiVersion();      // Returns 1
    int kernel = PhononNative.getKernelInUse();   // 2 = NEON, 3 = AVX2/SSE, 1 = Scalar
    String metrics = PhononNative.benchmarkGemm(4096, 1024, 200, 4);
}
```

---

## Native C ABI Exported Symbols

Shared library `libphonon2_android.so` provides the C ABI required by Fermion containers:

- `int phonon2_cpu_abi_version()`
- `int phonon2_cpu_kernel_in_use()`
- `int phonon2_cpu_set_threads(int threads)`
- `void* phonon2_cpu_create(int out_features, int in_features, void* sign_codes, void* is_hi, void* lo_scales, void* hi_scales, int mode)`
- `void phonon2_cpu_destroy(void* handle)`
- `void phonon2_cpu_dequant_row(void* handle, int row, float* dst)`
- `int phonon2_cpu_gemm(void* handle, const float* input, int m, float* output, int ldc)`

---

## License

Apache-2.0.
