"""
ser_bridge.py - SER 模型橋接模組

將 speech-emotion-recognition 專題訓練好的 WavLM 語音模型與 BERT 文字模型
整合到阿光的 Streamlit 應用中，實現完全本地化的雙模態情緒辨識。
"""

import io
import torch
import torchaudio
import streamlit as st
from pathlib import Path

from ser_model.model_def import SERModel
from ser_model.text_model import WhisperASR, TextEmotionClassifier
from ser_model.fusion import MultimodalSERSystem, LateFusion

# ── 常數 ──────────────────────────────────────────────
SAMPLE_RATE = 16000
MAX_DURATION_SEC = 10.0
MAX_WAVEFORM_LENGTH = int(SAMPLE_RATE * MAX_DURATION_SEC)  # 80000

MODEL_DIR = Path(__file__).parent / "ser_model"

# RAVDESS 情緒索引 → 英文名稱
IDX_TO_EMOTION = {
    0: "neutral", 1: "calm", 2: "happy", 3: "sad",
    4: "angry", 5: "fearful", 6: "disgust", 7: "surprised",
}

# RAVDESS 英文 → 阿光臨床中文標籤
RAVDESS_TO_AGUANG = {
    "neutral":   "正向平靜", "calm":      "放鬆", "happy":     "有活力",
    "sad":       "悲傷空虛", "angry":     "易怒煩躁", "fearful":   "焦慮煩躁",
    "disgust":   "身體緊繃", "surprised": "正向平靜",
}

# 阿光標籤 → 情緒分數
AGUANG_STRESS_HINT = {
    "正向平靜": 2, "放鬆": 2, "有活力": 1,
    "悲傷空虛": 7, "易怒煩躁": 7, "焦慮煩躁": 7,
    "身體緊繃": 6,
}

EMOTION_EMOJI = {
    "neutral": "😐", "calm": "😌", "happy": "😄", "sad": "😢",
    "angry": "😡", "fearful": "😨", "disgust": "🤢", "surprised": "😲",
}

# ── 模型載入（Streamlit 快取，整個 app 生命週期只載入一次）──
@st.cache_resource(show_spinner="🧠 正在啟動雙模態情緒引擎 (載入 WavLM, BERT, Whisper 首次約需 30 秒)...")
def load_ser_system():
    """載入訓練好的雙模態 SER 系統。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 載入 WavLM (Audio)
    audio_model = SERModel(freeze_wavlm=True)
    audio_weights_path = MODEL_DIR / "best_ser_model.pt"
    if audio_weights_path.exists():
        checkpoint = torch.load(audio_weights_path, map_location=device)
        audio_model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[SER Bridge] [OK] WavLM 模型已載入")
    else:
        print(f"[SER Bridge] [WARN] 找不到 WavLM 權重: {audio_weights_path}")
    audio_model.to(device)
    audio_model.eval()

    # 2. 載入 BERT (Text)
    text_classifier = TextEmotionClassifier(freeze_bert=True)
    text_weights_path = MODEL_DIR / "best_text_model.pt"
    if text_weights_path.exists():
        checkpoint = torch.load(text_weights_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        # 相容性處理：舊版 checkpoint 的 classifier 是單層 nn.Linear，
        # 新版是 nn.Sequential。若偵測到不匹配，只載入 BERT 主幹權重。
        try:
            text_classifier.load_state_dict(state_dict)
            print(f"[SER Bridge] [OK] BERT 模型已完整載入")
        except RuntimeError as load_err:
            print(f"[SER Bridge] [WARN] 分類頭架構不匹配，改用部分載入: {load_err}")
            text_classifier.load_state_dict(state_dict, strict=False)
            print(f"[SER Bridge] [OK] BERT 主幹已載入（分類頭使用隨機初始化）")
    else:
        print(f"[SER Bridge] [WARN] 找不到 BERT 權重: {text_weights_path}")
    text_classifier.to(device)
    text_classifier.eval()

    # 3. 載入 Whisper (ASR)
    whisper_asr = WhisperASR(model_name="base", device=device)
    print(f"[SER Bridge] [OK] Whisper ASR 已載入")

    # 4. 組合 Multimodal 系統
    fusion = LateFusion(audio_weight=0.95)  # 95% 語音, 5% 文字（消融實驗最佳權重）
    system = MultimodalSERSystem(
        audio_model=audio_model,
        text_classifier=text_classifier,
        whisper_asr=whisper_asr,
        fusion=fusion,
        device=device
    )

    return system


# ── 音訊前處理 ──────────────────────────────────────
def preprocess_audio_bytes(audio_bytes: bytes) -> torch.Tensor:
    import soundfile as sf
    audio_data, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
    waveform = torch.tensor(audio_data)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.t()

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    waveform = waveform.squeeze(0)

    if waveform.shape[0] > MAX_WAVEFORM_LENGTH:
        waveform = waveform[:MAX_WAVEFORM_LENGTH]
    elif waveform.shape[0] < MAX_WAVEFORM_LENGTH:
        waveform = torch.nn.functional.pad(
            waveform, (0, MAX_WAVEFORM_LENGTH - waveform.shape[0])
        )

    return waveform.unsqueeze(0)


# ── 推論 ────────────────────────────────────────────
@torch.no_grad()
def predict_emotion(audio_bytes: bytes) -> dict:
    """
    執行雙模態情緒辨識。
    """
    try:
        system = load_ser_system()
        waveform = preprocess_audio_bytes(audio_bytes)
        
        # Multimodal 系統內建 preprocess 與 predict，但我們自己先過前處理
        result = system.predict(waveform)
        
        ravdess_label = result["emotion"]
        confidence = result["confidence"]
        transcript = result["transcript"]
        prob_dict = result["probabilities"]

        aguang_label = RAVDESS_TO_AGUANG.get(ravdess_label, "正向平靜")
        
        # 讓壓力指數根據「信心度(confidence)」動態浮動，而不是寫死 7 分
        if ravdess_label in ["sad", "angry", "fearful", "disgust"]:
            # 負面情緒：基礎 5 分 + (信心度 * 5)
            stress_hint = min(10, int(5 + (confidence * 5)))
        else:
            # 正向/中性情緒：基礎 5 分 - (信心度 * 5)
            stress_hint = max(0, int(5 - (confidence * 5)))

        return {
            "ravdess_emotion": ravdess_label,
            "ravdess_emoji": EMOTION_EMOJI.get(ravdess_label, ""),
            "aguang_emotion": aguang_label,
            "confidence": round(confidence, 4),
            "stress_hint": stress_hint,
            "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            "transcript": transcript,
            "success": True,
        }

    except Exception as e:
        print(f"[SER Bridge] [ERROR] 推論失敗: {e}")
        import traceback
        traceback.print_exc()
        return {
            "ravdess_emotion": "unknown",
            "ravdess_emoji": "",
            "aguang_emotion": "",
            "confidence": 0.0,
            "stress_hint": 5,
            "probabilities": {},
            "transcript": "",
            "success": False,
            "error": str(e),
        }
