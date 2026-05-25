# services/ai_service.py
import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_BASE_DIR, ".env"), override=True)
MODEL = "llama-3.3-70b-versatile"
_client = None


def _get_client():
    global _client
    if _client is None and os.getenv("GROQ_API_KEY"):
        key = os.getenv("GROQ_API_KEY", "").strip()
        if key and key != "your_groq_api_key_here":
            _client = Groq(api_key=key)
    return _client


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _build_user_message(summary: dict) -> str:
    top_bull = summary.get("top_bullish", [])
    top_bear = summary.get("top_bearish", [])
    return f"""Stock market headline sentiment dataset:

Total headlines: {summary.get('total', 0)}
Bullish: {summary.get('bullish_count', 0)} ({summary.get('bullish_pct', 0)}%)
Bearish: {summary.get('bearish_count', 0)} ({summary.get('bearish_pct', 0)}%)
Neutral: {summary.get('neutral_count', 0)} ({summary.get('neutral_pct', 0)}%)
Average sentiment score (compound): {summary.get('avg_compound', 0)}
Market health score: {summary.get('market_health_score', 0)}/100
Overall signal: {summary.get('overall_signal', 'MIXED ⚪')}
Strong signals (|score| >= 0.5): {summary.get('strong_signals_count', 0)}

Top bullish headlines:
{chr(10).join('- ' + t for t in top_bull) or '- none'}

Top bearish headlines:
{chr(10).join('- ' + t for t in top_bear) or '- none'}
"""


def generate_insights(summary: dict) -> dict:
    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_groq_api_key_here":
        return {
            "insights": [
                {
                    "title": "API Key Required",
                    "body": "Add GROQ_API_KEY to your .env file. Get a free key at https://console.groq.com",
                    "type": "warning",
                    "icon": "🔑",
                }
            ],
            "executive_summary": "AI market insights require a Groq API key.",
            "market_call": "Configure GROQ_API_KEY to enable Wall Street-style analysis.",
            "risk_level": "MEDIUM",
        }

    system_prompt = """You are a senior Wall Street market analyst. Analyze stock market news
sentiment data and generate exactly 4 sharp actionable investment insights.
Return ONLY valid JSON, no markdown, no explanation:
{
  "insights": [
    {
      "title": "5 words max",
      "body": "2-3 sentences with exact numbers from data, use financial language",
      "type": "bullish|bearish|warning|opportunity",
      "icon": "relevant emoji"
    }
  ],
  "executive_summary": "2 sentences on overall market sentiment health",
  "market_call": "1 sentence overall signal e.g. Risk-on: Accumulate on dips",
  "risk_level": "LOW|MEDIUM|HIGH"
}"""

    user_message = _build_user_message(summary)
    client = _get_client()
    if not client:
        return {
            "insights": [
                {
                    "title": "API Key Required",
                    "body": "Groq client could not be initialized. Check GROQ_API_KEY in .env.",
                    "type": "warning",
                    "icon": "🔑",
                }
            ],
            "executive_summary": "AI insights unavailable.",
            "market_call": "Set a valid GROQ_API_KEY.",
            "risk_level": "MEDIUM",
        }

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            max_tokens=1200,
        )
        raw = response.choices[0].message.content or ""
        return _extract_json(raw)
    except json.JSONDecodeError:
        return {
            "insights": [
                {
                    "title": "Parse Error",
                    "body": "Could not parse AI JSON response. Try again.",
                    "type": "warning",
                    "icon": "⚠️",
                }
            ],
            "executive_summary": "AI response was not valid JSON.",
            "market_call": "Regenerate AI analysis.",
            "risk_level": "MEDIUM",
        }
    except Exception as e:
        return {
            "insights": [
                {
                    "title": "Groq API Error",
                    "body": str(e),
                    "type": "bearish",
                    "icon": "❌",
                }
            ],
            "executive_summary": "Failed to reach Groq API.",
            "market_call": "Verify API key and model availability.",
            "risk_level": "HIGH",
        }


def chat_with_data(question: str, summary: dict, history: list) -> str:
    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_groq_api_key_here":
        return "Add GROQ_API_KEY to your .env file to enable chat. Get a free key at https://console.groq.com"

    dataset_block = _build_user_message(summary)
    system_prompt = f"""You are an AI market analyst. Answer questions ONLY from this dataset summary:
{dataset_block}
Rules: 2-4 sentences max. Use financial language. Specific numbers only.
If outside data say: That specific data is not in your uploaded dataset."""

    messages = [{"role": "system", "content": system_prompt}]
    recent = history[-6:] if history else []
    for turn in recent:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})

    client = _get_client()
    if not client:
        return "Groq client not available. Check GROQ_API_KEY in .env."

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=400,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Sorry, I could not process that request: {e}"
