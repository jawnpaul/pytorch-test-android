package com.example.pytorchtest.utility

import java.util.*


fun String.toTwoDecimalPlaces(string: Float): String {
    return String.format(Locale.US, "%.2f", string)
}

fun String.toFPS(string: Float): String {
    return String.format(Locale.US, "%.1fFPS", string)
}

fun String.toMS(string: Long): String {
    return String.format(Locale.US, "%dms", string)
}

fun String.toAvgMS(string: Float): String {
    return String.format(Locale.US, "avg:%.0fms", string)
}



