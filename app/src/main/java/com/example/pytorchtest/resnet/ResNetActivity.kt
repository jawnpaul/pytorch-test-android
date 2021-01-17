package com.example.pytorchtest.resnet

import android.Manifest
import android.content.pm.PackageManager
import android.media.Image
import android.os.Bundle
import android.os.SystemClock
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.databinding.DataBindingUtil
import com.example.pytorchtest.R
import com.example.pytorchtest.databinding.ActivityResNetBinding
import com.example.pytorchtest.usecase.LuminosityAnalyzer
import com.example.pytorchtest.utility.*
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import timber.log.Timber
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

typealias LumaListener = (luma: Image?) -> Unit

class ResNetActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResNetBinding

    private var imageCapture: ImageCapture? = null

    private lateinit var cameraExecutor: ExecutorService

    private lateinit var utils: Utils
    private lateinit var constants: Constants

    private val TENSOR_HEIGHT: Int = 224
    private val TENSOR_WIDTH: Int = 224

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = DataBindingUtil.setContentView(this, R.layout.activity_res_net)

        utils = Utils()
        constants = Constants()

        cameraExecutor = Executors.newSingleThreadExecutor()
        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSIONS
            )
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.resNetViewFinder.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()
                .build()

            val imageAnalyzer = ImageAnalysis.Builder()
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, LuminosityAnalyzer { luma ->
                        //Timber.d("Average luminosity: $luma")
                        showResults(runAnalysisOnImage(luma))
                    })
                }


            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalyzer
                )

            } catch (exc: Exception) {
                Timber.e("Use case binding failed")
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults:
        IntArray
    ) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(
                    this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }

    private fun showResults(analysisResult: AnalysisResult) {
        runOnUiThread {

            binding.resNetResultOneTextview.text = analysisResult.topNClassNames[0]
            binding.resNetResultTwoTextview.text = analysisResult.topNClassNames[1]
            binding.resNetResultThreeTextview.text = analysisResult.topNClassNames[2]

            binding.resNetResultOneProbTextview.text =
                String().toTwoDecimalPlaces(analysisResult.topNScores[0])
            binding.resNetResultTwoProbTextview.text =
                String().toTwoDecimalPlaces(analysisResult.topNScores[1])
            binding.resNetResultThreeProbTextview.text =
                String().toTwoDecimalPlaces(analysisResult.topNScores[2])


            binding.resNetAnalysisDurationTextview.text =
                String().toMS(analysisResult.moduleForwardDuration)

            binding.resNetFramesPerSecondTextview.text =
                String().toFPS(1000f / analysisResult.analysisDuration)

        }

    }

    private fun runAnalysisOnImage(image: Image?): AnalysisResult {

        val moduleAssetName = File(utils.assetFilePath(this, "resnet18.pt"))
        val module = Module.load(moduleAssetName.toString())

        val inputTensorBuffer =
            Tensor.allocateFloatBuffer(3 * TENSOR_HEIGHT * TENSOR_WIDTH)
        val inputTensor = Tensor.fromBlob(
            inputTensorBuffer,
            longArrayOf(1, 3, TENSOR_HEIGHT.toLong(), TENSOR_WIDTH.toLong())
        )
        val startTime = SystemClock.elapsedRealtime()
        TensorImageUtils.imageYUV420CenterCropToFloatBuffer(
            image,
            0,
            TENSOR_WIDTH,
            TENSOR_HEIGHT,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB,
            inputTensorBuffer,
            0
        )

        val moduleForwardStartTime = SystemClock.elapsedRealtime()
        val outputTensor: Tensor = module.forward(IValue.from(inputTensor)).toTensor()
        val moduleForwardDuration: Long = SystemClock.elapsedRealtime() - moduleForwardStartTime

        val scores = outputTensor.dataAsFloatArray
        val ixs: IntArray =
            utils.topK(scores, 3)!!

        val topKClassNames = arrayOfNulls<String>(3)
        val topKScores = FloatArray(3)
        for (i in 0 until 3) {
            val ix = ixs[i]
            topKClassNames[i] = constants.IMAGENET_CLASSES[ix]
            topKScores[i] = scores[ix]
        }
        val analysisDuration = SystemClock.elapsedRealtime() - startTime

        return AnalysisResult(
            topKClassNames,
            topKScores,
            moduleForwardDuration,
            analysisDuration
        )
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}