plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.farhax0.ocrscanner"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.farhax0.ocrscanner"
        minSdk = 26
        targetSdk = 35
        versionCode = (System.getenv("GITHUB_RUN_NUMBER") ?: "1").toInt()
        versionName = "1.0." + (System.getenv("GITHUB_RUN_NUMBER") ?: "0")
    }

    signingConfigs {
        create("release") {
            // A fixed key so every new APK installs as an update over the previous one.
            // Override with your own private key via these environment variables / repo secrets.
            storeFile = file(System.getenv("OCR_KEYSTORE") ?: "../keystore/ocrscanner.jks")
            storePassword = System.getenv("OCR_KEYSTORE_PASSWORD") ?: "ocrscanner"
            keyAlias = System.getenv("OCR_KEY_ALIAS") ?: "ocrscanner"
            keyPassword = System.getenv("OCR_KEY_PASSWORD") ?: "ocrscanner"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("release")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.exifinterface:exifinterface:1.3.7")
    implementation("com.google.android.material:material:1.12.0")
    // On-device AI: automatic document detection, cropping, perspective correction & cleanup.
    implementation("com.google.android.gms:play-services-mlkit-document-scanner:16.0.0-beta1")
    // On-device AI text recognition (models bundled in the APK, works offline).
    implementation("com.google.mlkit:text-recognition:16.0.1")
    implementation("com.google.mlkit:text-recognition-devanagari:16.0.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-play-services:1.8.1")
}
