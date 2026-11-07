package com.farhax0.ocrscanner

import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import com.farhax0.ocrscanner.databinding.ActivityMainBinding
import com.google.mlkit.vision.documentscanner.GmsDocumentScannerOptions
import com.google.mlkit.vision.documentscanner.GmsDocumentScanning
import com.google.mlkit.vision.documentscanner.GmsDocumentScanningResult
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.devanagari.DevanagariTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var ui: ActivityMainBinding
    private var pages: List<PageResult> = emptyList()

    private val latin: TextRecognizer by lazy { TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS) }
    private val devanagari: TextRecognizer by lazy {
        TextRecognition.getClient(DevanagariTextRecognizerOptions.Builder().build())
    }

    /** Google's on-device document scanner: finds the page edges, crops, flattens and cleans it. */
    private val scanLauncher = registerForActivityResult(ActivityResultContracts.StartIntentSenderForResult()) { r ->
        if (r.resultCode == Activity.RESULT_OK) {
            val result = GmsDocumentScanningResult.fromActivityResultIntent(r.data)
            val uris = result?.pages?.map { it.imageUri }.orEmpty()
            if (uris.isNotEmpty()) process(uris)
        }
    }

    private val pickLauncher = registerForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris ->
        if (uris.isNotEmpty()) process(uris)
    }

    private val savePdfLauncher = registerForActivityResult(ActivityResultContracts.CreateDocument("application/pdf")) { uri ->
        uri?.let { saveTo(it) { SearchablePdf.build(pages) } }
    }

    private val saveTxtLauncher = registerForActivityResult(ActivityResultContracts.CreateDocument("text/plain")) { uri ->
        uri?.let { saveTo(it) { currentText().toByteArray() } }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ui = ActivityMainBinding.inflate(layoutInflater)
        setContentView(ui.root)

        ui.scanButton.setOnClickListener { startScanner() }
        ui.pickButton.setOnClickListener { pickLauncher.launch("image/*") }
        ui.copyButton.setOnClickListener {
            val cm = getSystemService(ClipboardManager::class.java)
            cm.setPrimaryClip(ClipData.newPlainText("OCR text", currentText()))
            toast("Text copied")
        }
        ui.shareButton.setOnClickListener { shareText() }
        ui.saveTxtButton.setOnClickListener { saveTxtLauncher.launch("scan-${stamp()}.txt") }
        ui.savePdfButton.setOnClickListener { savePdfLauncher.launch("scan-${stamp()}.pdf") }

        // Image shared into the app from the gallery or another app.
        if (intent?.action == Intent.ACTION_SEND) {
            @Suppress("DEPRECATION")
            val uri: Uri? = if (Build.VERSION.SDK_INT >= 33) {
                intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
            } else {
                intent.getParcelableExtra(Intent.EXTRA_STREAM)
            }
            uri?.let { process(listOf(it)) }
        }
    }

    private fun startScanner() {
        val options = GmsDocumentScannerOptions.Builder()
            .setScannerMode(GmsDocumentScannerOptions.SCANNER_MODE_FULL)
            .setGalleryImportAllowed(true)
            .setPageLimit(30)
            .setResultFormats(GmsDocumentScannerOptions.RESULT_FORMAT_JPEG)
            .build()
        GmsDocumentScanning.getClient(options)
            .getStartScanIntent(this)
            .addOnSuccessListener { scanLauncher.launch(IntentSenderRequest.Builder(it).build()) }
            .addOnFailureListener { e ->
                ui.status.text = "The automatic scanner needs Google Play services (${e.message}). " +
                    "You can still use \"Open images from gallery\"."
            }
    }

    private fun process(uris: List<Uri>) {
        val recognizer = if (ui.scriptDevanagari.isChecked) devanagari else latin
        setBusy(true)
        lifecycleScope.launch {
            try {
                val results = ArrayList<PageResult>()
                for ((i, uri) in uris.withIndex()) {
                    ui.status.text = "Reading page ${i + 1} of ${uris.size}…"
                    val bmp = withContext(Dispatchers.IO) { PageOcr.loadBitmap(this@MainActivity, uri) }
                    results.add(PageOcr.recognize(recognizer, bmp))
                }
                pages = results
                showResults()
            } catch (e: Exception) {
                ui.status.text = "Error: ${e.message}"
            } finally {
                setBusy(false)
            }
        }
    }

    private fun showResults() {
        val text = pages.joinToString("\n\n") { it.text.trimEnd() }
        ui.resultText.setText(text)
        val chars = pages.sumOf { p -> p.words.sumOf { it.text.length } }
        val conf = if (chars == 0) 0f else
            pages.sumOf { p -> (p.confidence * p.words.sumOf { it.text.length }).toDouble() }.toFloat() / chars
        val grade = when {
            chars == 0 -> "no text found"
            conf >= 90 -> "excellent"
            conf >= 80 -> "good"
            else -> "low - try better light or hold steadier"
        }
        ui.status.text = String.format(
            Locale.US, "%d page(s) · confidence %.1f%% (%s)", pages.size, conf, grade,
        )
        val unsure = pages.flatMap { it.uncertain }.map { it.text }
        if (unsure.isNotEmpty()) {
            ui.review.text = "Please check: " + unsure.take(40).joinToString(", ")
            ui.review.visibility = View.VISIBLE
        } else {
            ui.review.visibility = View.GONE
        }
        ui.actions.visibility = if (pages.isEmpty()) View.GONE else View.VISIBLE
    }

    private fun currentText(): String = ui.resultText.text?.toString().orEmpty()

    private fun shareText() {
        val dir = File(cacheDir, "share").apply { mkdirs() }
        val pdf = File(dir, "scan-${stamp()}.pdf")
        lifecycleScope.launch {
            withContext(Dispatchers.IO) { pdf.writeBytes(SearchablePdf.build(pages)) }
            val uri = FileProvider.getUriForFile(this@MainActivity, "$packageName.fileprovider", pdf)
            val send = Intent(Intent.ACTION_SEND).apply {
                type = "application/pdf"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_TEXT, currentText())
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(send, "Share scan"))
        }
    }

    private fun saveTo(uri: Uri, data: () -> ByteArray) {
        lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    contentResolver.openOutputStream(uri)?.use { it.write(data()) }
                }
                toast("Saved")
            } catch (e: Exception) {
                toast("Could not save: ${e.message}")
            }
        }
    }

    private fun setBusy(busy: Boolean) {
        ui.progress.visibility = if (busy) View.VISIBLE else View.GONE
        ui.scanButton.isEnabled = !busy
        ui.pickButton.isEnabled = !busy
    }

    private fun stamp() = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
}
