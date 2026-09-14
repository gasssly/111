"""
======================================================
 危機偵測安全網 (Safety Net)
======================================================
 在所有對話邏輯之上的最高優先級中斷器。
 不管信任分數多高、諮商進展到哪裡，
 只要偵測到危機訊號，立即中斷一切。

 危機等級：
   Level 0 (正常)：無異常
   Level 1 (留意)：出現低落暗示，加強關注
   Level 2 (警戒)：出現自傷暗示，切換為純傾聽
   Level 3 (危機)：明確自殺意念，立即中斷 + 通報
======================================================
"""


# ==================================================
# 危機關鍵字分級
# ==================================================

# Level 1：需要留意但不中斷
# 使用者可能只是在抱怨，不一定是真正的危機
LEVEL_1_KEYWORDS = {
    "活著好累", "好想消失", "不想面對", "逃避",
    "沒有意義", "什麼都不想做", "提不起勁",
    "好煩好煩", "受夠了", "厭世",
}

# Level 2：需要警戒，切換策略
# 出現自傷暗示，但還沒有明確的計畫
LEVEL_2_KEYWORDS = {
    "不想活", "想死", "死掉算了", "活不下去",
    "割", "自殘", "傷害自己", "自我了斷",
    "沒有人在乎", "世界沒有我會更好",
    "解脫", "一了百了",
}

# Level 3：立即中斷，最高危機
# 出現具體計畫或即時危險
LEVEL_3_KEYWORDS = {
    "我要去死", "我決定了", "再見了", "遺書",
    "跳下去", "吞藥", "割腕", "燒炭", "上吊",
    "今天就", "等一下就", "馬上就",
    "最後一次", "不會再",
}

# ==================================================
# 緊急回應模板
# ==================================================
CRISIS_RESPONSE = """我聽到你說的了，你現在一定非常痛苦。
你願意告訴我，是很勇敢的事。

但我現在很擔心你的安全。
請你現在就撥打以下電話，會有專業的人陪你：

  📞 安心專線：1925（24小時免費）
  📞 生命線：1995
  📞 張老師專線：1980

如果你身邊有人，請現在就告訴他你需要幫助。
我會一直在這裡陪你。"""


def check_safety(text: str) -> dict:
    """
    檢查單句文字的危機等級。

    回傳：
    {
        "level": 0~3,
        "triggered_words": [...],
        "action": "continue" / "monitor" / "caution" / "crisis_interrupt",
        "crisis_response": None 或 緊急回應文字
    }
    """
    # Level 3 最先檢查（最高優先）
    l3_hits = [w for w in LEVEL_3_KEYWORDS if w in text]
    if l3_hits:
        return {
            "level": 3,
            "triggered_words": l3_hits,
            "action": "crisis_interrupt",
            "crisis_response": CRISIS_RESPONSE,
        }

    # Level 2
    l2_hits = [w for w in LEVEL_2_KEYWORDS if w in text]
    if l2_hits:
        return {
            "level": 2,
            "triggered_words": l2_hits,
            "action": "caution",
            "crisis_response": None,
        }

    # Level 1
    l1_hits = [w for w in LEVEL_1_KEYWORDS if w in text]
    if l1_hits:
        return {
            "level": 1,
            "triggered_words": l1_hits,
            "action": "monitor",
            "crisis_response": None,
        }

    return {
        "level": 0,
        "triggered_words": [],
        "action": "continue",
        "crisis_response": None,
    }
