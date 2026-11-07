package com.farhax0.ocrscanner

import android.graphics.Bitmap
import java.io.ByteArrayOutputStream
import java.nio.charset.Charset
import java.util.Locale

/**
 * Writes a searchable PDF: each page is the scanned image with the
 * recognised words laid over it as invisible text, so the PDF can be
 * searched, selected and copied in any PDF viewer.
 */
object SearchablePdf {
    private val cp1252: Charset = Charset.forName("windows-1252")
    private const val A4_WIDTH_IN = 8.27f

    /** The built-in PDF font covers Latin text; other scripts get image-only pages. */
    fun canEmbedText(pages: List<PageResult>): Boolean {
        val enc = cp1252.newEncoder()
        return pages.all { p -> p.words.all { enc.canEncode(it.text) } }
    }

    fun build(pages: List<PageResult>): ByteArray {
        val withText = canEmbedText(pages)
        val objects = ArrayList<ByteArray>()
        // 1 catalog, 2 pages, 3 font; then 3 objects per page (page, image, content).
        val pageIds = pages.indices.map { 4 + it * 3 }
        objects.add("<< /Type /Catalog /Pages 2 0 R >>".toByteArray())
        objects.add("<< /Type /Pages /Kids [${pageIds.joinToString(" ") { "$it 0 R" }}] /Count ${pages.size} >>".toByteArray())
        objects.add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>".toByteArray())

        for ((i, page) in pages.withIndex()) {
            val bmp = page.bitmap
            val jpg = ByteArrayOutputStream().also { bmp.compress(Bitmap.CompressFormat.JPEG, 85, it) }.toByteArray()
            val dpi = maxOf(72f, bmp.width / A4_WIDTH_IN)
            val s = 72f / dpi
            val pw = bmp.width * s
            val ph = bmp.height * s
            val pageId = pageIds[i]
            val content = ByteArrayOutputStream()
            content.write(f("q %.2f 0 0 %.2f 0 0 cm /Im0 Do Q\n", pw, ph))
            if (withText) {
                content.write("BT 3 Tr\n".toByteArray())
                for (w in page.words) {
                    val bw = w.box.width()
                    val bh = w.box.height()
                    if (bw <= 0 || bh <= 0 || w.text.isEmpty()) continue
                    val size = maxOf(1f, bh * s)
                    val natural = 0.5f * size * w.text.length
                    val tz = 100f * (bw * s) / natural
                    val baseline = ph - (w.box.top + bh) * s + 0.2f * size
                    content.write(f("/F1 %.2f Tf %.1f Tz 1 0 0 1 %.2f %.2f Tm (", size, tz, w.box.left * s, baseline))
                    content.write(escape((w.text + " ").toByteArray(cp1252)))
                    content.write(") Tj\n".toByteArray())
                }
                content.write("ET\n".toByteArray())
            }
            val c = content.toByteArray()
            objects.add(
                f(
                    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.2f %.2f] " +
                        "/Resources << /XObject << /Im0 %d 0 R >> /Font << /F1 3 0 R >> >> /Contents %d 0 R >>",
                    pw, ph, pageId + 1, pageId + 2,
                )
            )
            objects.add(
                f(
                    "<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB " +
                        "/BitsPerComponent 8 /Filter /DCTDecode /Length %d >>\nstream\n",
                    bmp.width, bmp.height, jpg.size,
                ) + jpg + "\nendstream".toByteArray()
            )
            objects.add("<< /Length ${c.size} >>\nstream\n".toByteArray() + c + "endstream".toByteArray())
        }

        val out = ByteArrayOutputStream()
        out.write("%PDF-1.4\n%âãÏÓ\n".toByteArray(Charsets.ISO_8859_1))
        val offsets = ArrayList<Int>()
        objects.forEachIndexed { idx, obj ->
            offsets.add(out.size())
            out.write("${idx + 1} 0 obj\n".toByteArray())
            out.write(obj)
            out.write("\nendobj\n".toByteArray())
        }
        val xref = out.size()
        out.write("xref\n0 ${objects.size + 1}\n0000000000 65535 f \n".toByteArray())
        for (o in offsets) out.write(String.format(Locale.US, "%010d 00000 n \n", o).toByteArray())
        out.write("trailer\n<< /Size ${objects.size + 1} /Root 1 0 R >>\nstartxref\n$xref\n%%EOF\n".toByteArray())
        return out.toByteArray()
    }

    private fun f(fmt: String, vararg args: Any): ByteArray = String.format(Locale.US, fmt, *args).toByteArray()

    private fun escape(b: ByteArray): ByteArray {
        val out = ByteArrayOutputStream()
        for (x in b) {
            if (x == '\\'.code.toByte() || x == '('.code.toByte() || x == ')'.code.toByte()) out.write('\\'.code)
            out.write(x.toInt())
        }
        return out.toByteArray()
    }
}
