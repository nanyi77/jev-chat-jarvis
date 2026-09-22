package com.jev.probe.jev

import android.util.Log
import com.jev.probe.core.Analysis
import com.jev.probe.core.ChatSnapshot
import com.jev.probe.core.Choice
import com.jev.probe.core.Prefs
import com.jev.probe.core.RankedReply
import com.jev.probe.core.Score
import com.jev.probe.core.kb.ChatContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * The Jev judgment route only: the 7 judgment questions in one call, and the
 * ranking question over already-drafted candidates. Reads judgeProvider /
 * judgeBaseUrl / judgeKey / judgeModel from [Prefs]; nothing generative here.
 */
class JudgeClient(private val prefs: Prefs) {

    /**
     * The 7 judgment questions (fast, ~1s). Errors are returned, not thrown.
     *
     * @param ctx D-stage knowledge context; null or empty means the request body
     *        is byte-for-byte what v1.2 sent.
     */
    fun judge(snapshot: ChatSnapshot, relationship: String, ctx: ChatContext? = null): Analysis {
        val start = System.currentTimeMillis()
        return try {
            val answers = postDecisions(
                snapshot, relationship, ctx,
                JevQuestions.judge()
            )
            Analysis(
                trueIntent = parseChoice(answers.optJSONObject("true_intent")),
                dangerLevel = parseScore(answers.optJSONObject("danger_level")),
                sheNeeds = parseChoice(answers.optJSONObject("she_needs")),
                shouldReplyNow = answers.optJSONObject("should_reply_now")?.optDouble("noul"),
                bestAction = parseChoice(answers.optJSONObject("best_action")),
                tensionResolved = answers.optJSONObject("tension_resolved")?.optDouble("noul"),
                literalQuestion = answers.optJSONObject("literal_question")?.optDouble("noul"),
                rankedReplies = emptyList(),
                latencyMs = System.currentTimeMillis() - start
            )
        } catch (e: Exception) {
            Log.w(TAG, "judge failed: ${e.message}")
            Analysis(null, null, null, null, null, null, null, emptyList(),
                System.currentTimeMillis() - start, error = e.message ?: "判断接口请求失败")
        }
    }

    /** Ask Jev which of the candidate replies is best; throws on failure. */
    fun rank(
        snapshot: ChatSnapshot,
        relationship: String,
        candidates: List<String>,
        ctx: ChatContext? = null
    ): List<RankedReply> {
        val questions = JSONObject().put("best_reply",
            JevQuestions.rankQuestion(candidates).getJSONObject("best_reply"))
        val answers = postDecisions(snapshot, relationship, ctx, questions)
        return parseRanked(answers.optJSONObject("best_reply"), candidates)
    }

    /**
     * POST one decisions request, with the knowledge fields when there are any.
     *
     * Defensive retry: whether the live `alpha/decisions` endpoint accepts the
     * new `background` / `history` state fields or rejects unknown ones with a
     * 4xx is not verified against production yet (see the A-stage report). If a
     * request carrying them comes back 4xx, it is sent again once without them,
     * so an unverified field can degrade the analysis but never break it.
     */
    private fun postDecisions(
        snapshot: ChatSnapshot,
        relationship: String,
        ctx: ChatContext?,
        questions: JSONObject
    ): JSONObject {
        val background = ctx?.background(relationship) ?: ""
        val history = ctx?.history ?: emptyList()
        val enriched = background.isNotBlank() || history.isNotEmpty()
        return try {
            send(JevQuestions.buildState(snapshot, relationship, background, history), questions)
        } catch (e: ApiException) {
            // Chat completions 4xx is a real rejection (key, model, body), not an
            // unknown decisions-API field. Only the decisions route retries plain.
            if (!prefs.usesChatJudge() && enriched && e.status != null && e.status in 400..499) {
                Log.w(TAG, "judge HTTP ${e.status} with background/history; retrying plain")
                send(JevQuestions.buildState(snapshot, relationship), questions)
            } else throw e
        }
    }

    private fun send(state: JSONObject, questions: JSONObject): JSONObject {
        if (prefs.usesChatJudge()) return sendChatJudge(state, questions)
        val url = prefs.judgeEndpoint()
        val body = JSONObject()
            .put("model", prefs.judgeModel)
            .put("state", state)
            .put("questions", questions)
        val resp = HttpJson.post(url, prefs.judgeKey, body, Route.JUDGE, HttpJson.headersFor(url))
        return resp.optJSONObject("answers") ?: JSONObject()
    }

    /**
     * DeepSeek official API has no `/alpha/decisions`. Ask `deepseek-flash` for
     * the same answer object the decisions parser already understands.
     * JSON mode can return an empty content; try once more before failing.
     */
    private fun sendChatJudge(state: JSONObject, questions: JSONObject): JSONObject {
        var last = ""
        repeat(2) {
            last = postChatJudge(state, questions)
            val parsed = extractObject(last) ?: return@repeat
            return normalizeAnswers(unwrapAnswers(parsed, questions), questions)
        }
        throw ApiException(Route.JUDGE, null, "没有返回合法 JSON：${last.take(80).ifBlank { "响应为空" }}")
    }

    private fun postChatJudge(state: JSONObject, questions: JSONObject): String {
        val url = prefs.judgeEndpoint()
        val messages = JSONArray()
            .put(JSONObject().put("role", "system").put("content", CHAT_JUDGE_SYSTEM))
            .put(JSONObject().put("role", "user").put("content",
                "state:\n$state\n\nquestions:\n$questions\n\n按格式输出 JSON。"))
        val body = JSONObject()
            .put("model", prefs.modelFor(url, prefs.judgeModel))
            .put("messages", messages)
            .put("temperature", 0.2)
            .put("max_tokens", 1600)
        if (url.contains("api.deepseek.com", ignoreCase = true)) {
            body.put("response_format", JSONObject().put("type", "json_object"))
        }
        val resp = HttpJson.post(url, prefs.judgeKey, body, Route.JUDGE, HttpJson.headersFor(url))
        val message = resp.optJSONArray("choices")?.optJSONObject(0)?.optJSONObject("message")
        val content = message?.optString("content").orEmpty()
        if (content.isNotBlank()) return content
        return message?.optString("reasoning_content").orEmpty()
    }

    private fun unwrapAnswers(raw: JSONObject, questions: JSONObject): JSONObject {
        val answers = raw.optJSONObject("answers") ?: return raw
        val keys = questions.keys()
        while (keys.hasNext()) {
            if (answers.has(keys.next())) return answers
        }
        return raw
    }

    private fun extractObject(content: String): JSONObject? {
        val start = content.indexOf('{')
        val end = content.lastIndexOf('}')
        if (start < 0 || end <= start) return null
        return try {
            JSONObject(content.substring(start, end + 1))
        } catch (_: Exception) {
            null
        }
    }

    private fun normalizeAnswers(raw: JSONObject, questions: JSONObject): JSONObject {
        val out = JSONObject()
        val keys = questions.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            val q = questions.optJSONObject(key) ?: continue
            val value = if (raw.has(key) && !raw.isNull(key)) raw.opt(key) else null
            out.put(key, when (q.optString("type")) {
                "noul" -> normalizeNoul(value)
                "score" -> normalizeScore(value, q.optJSONArray("criteria"))
                else -> normalizeChoice(value, q.optJSONObject("criteria"), key == "best_reply")
            })
        }
        return out
    }

    private fun normalizeNoul(value: Any?): JSONObject {
        val n = when (value) {
            is JSONObject -> when {
                value.has("noul") -> value.optDouble("noul", 0.5)
                value.has("value") -> boolish(value.opt("value"), value.optDouble("confidence", 0.5))
                value.has("answer") -> boolish(value.opt("answer"), 0.5)
                else -> 0.5
            }
            else -> boolish(value, 0.5)
        }.coerceIn(0.0, 1.0)
        return JSONObject().put("noul", n)
    }

    private fun boolish(value: Any?, fallback: Double): Double = when (value) {
        null, JSONObject.NULL -> fallback
        is Boolean -> if (value) 1.0 else 0.0
        is Number -> value.toDouble()
        is String -> when {
            value.equals("true", true) -> 1.0
            value.equals("false", true) -> 0.0
            else -> value.toDoubleOrNull() ?: fallback
        }
        else -> fallback
    }

    private fun normalizeScore(value: Any?, criteria: JSONArray?): JSONObject {
        val max = ((criteria?.length() ?: 10) - 1).coerceAtLeast(0)
        val obj = value as? JSONObject
        val score = when (value) {
            is JSONObject -> value.optDouble("score", 0.0)
            is Number -> value.toDouble()
            is String -> value.toDoubleOrNull() ?: 0.0
            else -> 0.0
        }.coerceIn(0.0, max.toDouble())
        val conf = (obj?.optDouble("confidence", 0.5) ?: 0.5).coerceIn(0.0, 1.0)
        return JSONObject().put("score", score).put("confidence", conf)
    }

    private fun normalizeChoice(value: Any?, criteria: JSONObject?, withProbs: Boolean): JSONObject {
        val allowed = ArrayList<String>()
        criteria?.keys()?.forEach { allowed.add(it) }
        var choice = when (value) {
            is JSONObject -> value.optString("choice")
            is String -> value
            else -> ""
        }
        if (allowed.isNotEmpty() && choice !in allowed) {
            val hit = allowed.firstOrNull { it.equals(choice, true) }
            if (hit != null) choice = hit
        }
        val obj = value as? JSONObject
        val conf = (if (obj != null && obj.has("confidence")) obj.optDouble("confidence", 0.5) else 0.5)
            .coerceIn(0.0, 1.0)
        val out = JSONObject().put("choice", choice).put("confidence", conf)
        if (withProbs && allowed.isNotEmpty()) {
            val given = obj?.optJSONObject("probabilities")
            val probs = JSONObject()
            var sum = 0.0
            var complete = given != null
            for (k in allowed) {
                if (given == null || !given.has(k)) {
                    complete = false
                    break
                }
                val p = given.optDouble(k, 0.0)
                probs.put(k, p)
                sum += p
            }
            if (!complete || sum <= 0.0) {
                val others = (allowed.size - 1).coerceAtLeast(1)
                val rest = ((1.0 - conf) / others).coerceAtLeast(0.0)
                for (k in allowed) probs.put(k, if (k == choice) conf else rest)
            }
            out.put("probabilities", probs)
        }
        return out
    }

    private fun parseChoice(o: JSONObject?): Choice? {
        o ?: return null
        val probs = HashMap<String, Double>()
        o.optJSONObject("probabilities")?.let { p ->
            p.keys().forEach { k -> probs[k] = p.optDouble(k) }
        }
        return Choice(o.optString("choice"), o.optDouble("confidence", 0.0), probs)
    }

    private fun parseScore(o: JSONObject?): Score? {
        o ?: return null
        val legend = o.optJSONObject("legend")
        val maxLevel = legend?.keys()?.asSequence()?.mapNotNull { it.toIntOrNull() }?.maxOrNull() ?: 9
        return Score(o.optDouble("score", 0.0), o.optDouble("confidence", 0.0), maxLevel)
    }

    private fun parseRanked(o: JSONObject?, candidates: List<String>): List<RankedReply> {
        val keys = listOf("reply_a", "reply_b", "reply_c")
        val probs = o?.optJSONObject("probabilities")
        val list = candidates.mapIndexed { i, text ->
            RankedReply(text, probs?.optDouble(keys.getOrElse(i) { "" }, 0.0) ?: 0.0)
        }
        return list.sortedByDescending { it.prob }
    }

    companion object {
        private const val TAG = "JEVASSIST"
        private const val CHAT_JUDGE_SYSTEM =
            "你是对话判断器。只输出一个 JSON 对象，不要 markdown，不要解释。" +
                "键必须与 questions 的键完全一致。" +
                "type=noul 的题输出 {\"noul\": 0到1的小数}，1 表示 criteria.true，0 表示 criteria.false。" +
                "type=choice 的题输出 {\"choice\":\"criteria 的某个键\",\"confidence\":0到1}，choice 必须是键本身。" +
                "type=score 的题输出 {\"score\":整数,\"confidence\":0到1}，0 是 criteria 数组第一档，最大是数组长度减 1。" +
                "如果有 best_reply，再加 probabilities，键为 reply_a、reply_b、reply_c，和为 1，最大的那个必须等于 choice。" +
                "示例：{\"true_intent\":{\"choice\":\"casual_chat\",\"confidence\":0.8},\"danger_level\":{\"score\":1,\"confidence\":0.7},\"should_reply_now\":{\"noul\":0.2}}"
    }
}
