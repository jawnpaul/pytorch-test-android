package com.example.pytorchtest.utility

data class AnalysisResult(
    val topNClassNames: Array<String?>,
    val topNScores: FloatArray,
    val moduleForwardDuration: Long,
    val analysisDuration: Long
) {
}