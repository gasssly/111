"""
ser_bridge.py - 雲端展示版 (Cloud Mock Version)

因為本地的 WavLM 與 BERT 雙模態模型檔案高達 800MB，
超過了 GitHub 上傳限制與 Streamlit Cloud 免費伺服器的 1GB 記憶體限制，
如果在雲端強行載入會導致伺服器崩潰 (Out of Memory)。

為了讓阿光能順利在網頁上 Demo 給教授看，
這個版本是「雲端安全版」，拔除了 PyTorch 大型模型的載入，
當有人按麥克風時，會回傳固定的預設值並提示。
"""

import streamlit as st

# RAVDESS 英文 → 阿光臨床中文標籤 (與原本一致)
RAVDESS_TO_AGUANG = {
    "neutral":   "正向平靜", "calm":      "放鬆", "happy":     "有活力",
    "sad":       "悲傷空虛", "angry":     "易怒煩躁", "fearful":   "焦慮煩躁",
    "disgust":   "身體緊繃", "surprised": "正向平靜",
}

# ── 推論 (Mock) ────────────────────────────────────────────
def predict_emotion(audio_bytes: bytes) -> dict:
    """
    雲端展示版的 Mock 推論，不消耗任何伺服器資源。
    """
    return {
        "ravdess_emotion": "calm",
        "ravdess_emoji": "😌",
        "aguang_emotion": "放鬆",
        "confidence": 0.99,
        "stress_hint": 1,
        "probabilities": {"calm": 0.99, "happy": 0.01},
        "transcript": "【阿光雲端廣播】為了網頁順暢，展示版已關閉超大型語音情緒模型。請您直接打字與阿光聊天喔！",
        "success": True,
    }
