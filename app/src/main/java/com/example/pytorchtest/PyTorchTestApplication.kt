package com.example.pytorchtest

import android.app.Application
import androidx.appcompat.app.AppCompatDelegate
import timber.log.Timber

class PyTorchTestApplication : Application() {

    override fun onCreate() {
        super.onCreate()
        AppCompatDelegate.setDefaultNightMode(
            AppCompatDelegate.MODE_NIGHT_NO
        )
        if (BuildConfig.DEBUG) {
            Timber.plant(Timber.DebugTree())
        }

    }
}