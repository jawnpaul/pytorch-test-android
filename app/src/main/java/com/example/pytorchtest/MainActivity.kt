package com.example.pytorchtest

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.databinding.DataBindingUtil
import com.example.pytorchtest.databinding.ActivityMainBinding
import com.example.pytorchtest.mobilenet.MobileNetActivity
import com.example.pytorchtest.resnet.ResNetActivity

private lateinit var binding: ActivityMainBinding

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = DataBindingUtil.setContentView(this, R.layout.activity_main)

        binding.mobileNetCardview.setOnClickListener {
            val intent = Intent(this, MobileNetActivity::class.java)
            startActivity(intent)
        }

        binding.resNetCardview.setOnClickListener {
            val intent = Intent(this, ResNetActivity::class.java)
            startActivity(intent)
        }
    }
}