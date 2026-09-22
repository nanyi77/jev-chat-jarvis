package com.jev.probe.jev

import android.graphics.Bitmap
import android.util.Base64
import com.jev.probe.core.Prefs
import java.io.ByteArrayOutputStream
import org.json.JSONArray
import org.json.JSONObject

/**
 * The vision route: an OpenAI-compatible `/chat/completions` endpoint that
 * accepts `image_url` content parts. A-stage shell only — B stage wires it to
 * the screenshot pipeline (see docs/v1.3-plan.md "OCR 分层").
 *
 * Reads visionBaseUrl / visionKey / visionModel from [Prefs]. The base URL does
 * not inherit from a custom reply host; the key still falls back reply -> judge.
 * DeepSeek official `deepseek-flash` accepts images. `deepseek-v4-pro` does not.
 *
 * Wire format notes that cost real debugging time:
 * - JPEG, not PNG: a screenshot as PNG base64 is several times larger.
 * - `Base64.NO_WRAP`: Android's default inserts newlines, which corrupts the
 *   data URL.
 * - DashScope's compatible-mode wants the image part before the text part.
 *   DeepSeek's documented order is text, then image.
 */
class VisionClient(private val prefs: Prefs) {

    /**
     * Send a screenshot and get the transcribed dialog back as plain text.
     * B stage will parse this into bubbles; A stage only proves the route works.
     *
     * @param imageBase64Jpeg base64 of a JPEG, without the `data:` prefix.
     */
    fun extractDialog(imageBase64Jpeg: String): String = ask(
        imageBase64Jpeg,
        "你是聊天截图转写助手。把图中聊天气泡按从上到下的顺序转写成文本，" +
            "每行一条，格式 `我：正文` 或 `对方：正文`。只输出转写结果，不要解释。"
    )

    /** Generic single-question call against the image (used by the settings test). */
    fun ask(imageBase64Jpeg: String, prompt: String): String {
        val url = prefs.visionEndpoint()
        val image = JSONObject()
            .put("type", "image_url")
            .put("image_url", JSONObject().put("url", "data:image/jpeg;base64,$imageBase64Jpeg"))
        val text = JSONObject().put("type", "text").put("text", prompt)
        // DashScope rejects text-then-image. DeepSeek's docs use text, then image.
        val content = JSONArray()
        if (url.contains("dashscope", ignoreCase = true)) {
            content.put(image).put(text)
        } else {
            content.put(text).put(image)
        }
        val messages = JSONArray().put(
            JSONObject().put("role", "user").put("content", content))
        val body = JSONObject()
            .put("model", prefs.modelFor(url, prefs.visionModel))
            .put("messages", messages)
            .put("temperature", 0.0)
            .put("max_tokens", 1500)
        val resp = HttpJson.post(url, prefs.effectiveVisionKey(), body, Route.VISION, HttpJson.headersFor(url))
        return resp.optJSONArray("choices")?.optJSONObject(0)
            ?.optJSONObject("message")?.optString("content") ?: ""
    }

    companion object {
        /** Bitmap -> JPEG base64 in the exact form [ask] expects. */
        fun encodeJpeg(bitmap: Bitmap, quality: Int = 80): String {
            val out = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, quality, out)
            return Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP)
        }

        /**
         * Official DeepSeek vision is `deepseek-flash` (and the retired flash
         * aliases). `deepseek-v4-pro` rejects image parts.
         */
        fun supportsVision(baseUrl: String, model: String = ""): Boolean {
            if (!baseUrl.contains("api.deepseek.com", ignoreCase = true)) return true
            val m = model.trim().lowercase()
            if (m.isEmpty() || m.contains("flash")) return true
            return false
        }
    }
}
