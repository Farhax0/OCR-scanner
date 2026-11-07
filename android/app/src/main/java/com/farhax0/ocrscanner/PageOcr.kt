package com.farhax0.ocrscanner

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognizer
import kotlinx.coroutines.tasks.await

/** One recognised word with its position in the page bitmap. */
data class OcrWord(val text: String, val confidence: Float, val box: Rect)

/** Result for one page. */
class PageResult(
    val bitmap: Bitmap,
    val text: String,
    val words: List<OcrWord>,
) {
    /** Character-weighted mean confidence, 0..100. */
    val confidence: Float
        get() {
            val n = words.sumOf { it.text.length }
            if (n == 0) return 0f
            return words.sumOf { (it.text.length * it.confidence).toDouble() }.toFloat() / n * 100f
        }

    val uncertain: List<OcrWord> get() = words.filter { it.confidence < LOW_CONFIDENCE }

    companion object {
        const val LOW_CONFIDENCE = 0.6f
    }
}

object PageOcr {
    /** Longest side used for OCR and the PDF: enough for small print, bounded memory. */
    private const val MAX_SIDE = 4000

    /** Decode an image with its EXIF rotation applied, downscaled to at most MAX_SIDE. */
    fun loadBitmap(context: Context, uri: Uri): Bitmap {
        val resolver = context.contentResolver
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        resolver.openInputStream(uri).use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= MAX_SIDE) sample *= 2
        val opts = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        var bmp = resolver.openInputStream(uri).use { BitmapFactory.decodeStream(it, null, opts) }
            ?: throw IllegalArgumentException("Cannot read image")
        val rotation = try {
            resolver.openInputStream(uri).use { ExifInterface(it!!).rotationDegrees }
        } catch (e: Exception) {
            0
        }
        if (rotation != 0) {
            val m = Matrix().apply { postRotate(rotation.toFloat()) }
            bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
        }
        val scale = MAX_SIDE.toFloat() / maxOf(bmp.width, bmp.height)
        if (scale < 1f) {
            bmp = Bitmap.createScaledBitmap(bmp, (bmp.width * scale).toInt(), (bmp.height * scale).toInt(), true)
        }
        return bmp
    }

    suspend fun recognize(recognizer: TextRecognizer, bitmap: Bitmap): PageResult {
        val result: Text = recognizer.process(InputImage.fromBitmap(bitmap, 0)).await()
        val words = ArrayList<OcrWord>()
        val out = StringBuilder()
        for (block in sortBlocks(result.textBlocks)) {
            if (out.isNotEmpty()) out.append("\n\n")
            block.lines.forEachIndexed { i, line ->
                if (i > 0) out.append('\n')
                out.append(line.elements.joinToString(" ") { it.text })
                for (el in line.elements) {
                    val box = el.boundingBox ?: continue
                    words.add(OcrWord(el.text, el.confidence, box))
                }
            }
        }
        return PageResult(bitmap, out.toString().trim() + "\n", words)
    }

    /**
     * Reading order: top-to-bottom, but blocks side by side (columns) are
     * kept left-to-right when they start at about the same height.
     */
    private fun sortBlocks(blocks: List<Text.TextBlock>): List<Text.TextBlock> {
        val withBox = blocks.filter { it.boundingBox != null }
        val tol = (withBox.map { it.boundingBox!!.height() / maxOf(1, it.lines.size) }.average().takeIf { !it.isNaN() } ?: 20.0)
        return withBox.sortedWith { a, b ->
            val ra = a.boundingBox!!
            val rb = b.boundingBox!!
            if (kotlin.math.abs(ra.top - rb.top) < tol) ra.left.compareTo(rb.left) else ra.top.compareTo(rb.top)
        } + blocks.filter { it.boundingBox == null }
    }
}
