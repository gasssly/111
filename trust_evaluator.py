"""
======================================================
 信任度量化評估器 (Trust Evaluator)
======================================================
 純 Library 版本：供 main.py import 使用。
 只保留 compute_trust_score() 與 build_trust_prompt() 函式。

 核心演算法：
   指標 1：回覆字數 (滿分 40) — 權重最高，因為「願不願意多說」是最直覺的信任指標
   指標 2：情緒詞密度 (滿分 30) — 偵測使用者是否暴露內心感受
   指標 3：自我揭露程度 (滿分 30) — 偵測使用者是否主動提供未被問及的個人細節

 理論基礎：
   - Jourard (1971) 自我揭露理論
   - Pennebaker (2001) 情緒書寫理論
======================================================
"""

# ============================================================
# 繁體中文情緒詞庫
# ============================================================
EMOTION_WORDS = {
    # ---- 負面脆弱類 (願意示弱 = 高信任) ----
    "難過", "傷心", "心情不好", "低落", "沮喪", "憂鬱", "鬱悶",
    "焦慮", "緊張", "害怕", "恐懼", "不安", "擔心", "煩惱",
    "生氣", "憤怒", "煩", "煩躁", "受不了", "崩潰", "無力",
    "孤單", "寂寞", "空虛", "迷茫", "無助", "絕望",
    "壓力", "累", "好累", "疲憊", "撐不住", "想哭", "哭",
    "後悔", "自責", "愧疚", "丟臉", "尷尬",
    "委屈", "不甘心", "心酸", "心痛", "受傷",
    "討厭", "噁心", "厭煩", "不想",
    # ---- 正面坦誠類 ----
    "開心", "高興", "快樂", "幸福", "感動", "溫暖",
    "感謝", "謝謝", "感激", "安心", "放鬆", "舒服",
    "期待", "興奮", "喜歡", "愛", "珍惜",
    # ---- 內心獨白標記詞 ----
    "其實", "老實說", "說實話", "坦白講", "不瞞你說",
    "我覺得", "我感覺", "我認為", "我心裡", "我內心",
}

# ============================================================
# 自我揭露關鍵字
# ============================================================
SELF_DISCLOSURE_KEYWORDS = {
    # ---- 家庭關係 ----
    "爸", "媽", "爸爸", "媽媽", "父親", "母親", "爸媽", "父母",
    "哥", "姐", "弟", "妹", "哥哥", "姐姐", "弟弟", "妹妹",
    "兒子", "女兒", "孫子", "孫女", "小孩", "孩子",
    "老公", "老婆", "先生", "太太", "另一半",
    "阿公", "阿嬤", "爺爺", "奶奶", "外公", "外婆",
    "家人", "家裡",
    # ---- 感情關係 ----
    "女朋友", "男朋友", "對象", "喜歡的人", "暗戀", "前任", "分手",
    "交往", "感情", "戀愛",
    # ---- 個人經歷 / 隱私 ----
    "小時候", "以前", "那時候", "曾經", "我記得",
    "工作", "同事", "主管", "老闆", "薪水", "加班",
    "學校", "同學", "朋友", "室友",
    "看醫生", "吃藥", "失眠", "住院",
}


# ============================================================
# 信任度計算引擎
# ============================================================
def compute_trust_score(user_turns: list) -> dict:
    """
    輸入使用者的所有歷史發言（list of str），
    輸出信任度評估結果。
    """
    if not user_turns:
        return {"total_score": 0, "phase": "Phase 0",
                "length_score": 0, "emotion_score": 0, "disclosure_score": 0,
                "emotion_hits": [], "disclosure_hits": []}

    # --- 指標 1：回覆字數 (滿分 40) ---
    avg_len = sum(len(t) for t in user_turns) / len(user_turns)
    if avg_len <= 5:
        length_score = 0
    elif avg_len <= 15:
        length_score = 10
    elif avg_len <= 30:
        length_score = 20
    elif avg_len <= 60:
        length_score = 30
    else:
        length_score = 40

    # --- 指標 2：情緒詞彙密度 (滿分 30) ---
    all_emotion_hits = []
    for text in user_turns:
        for word in EMOTION_WORDS:
            if word in text:
                all_emotion_hits.append(word)

    emotion_count = len(all_emotion_hits)
    if emotion_count == 0:
        emotion_score = 0
    elif emotion_count <= 2:
        emotion_score = 10
    elif emotion_count <= 5:
        emotion_score = 20
    else:
        emotion_score = 30

    # --- 指標 3：自我揭露程度 (滿分 30) ---
    all_disclosure_hits = []
    for text in user_turns:
        for word in SELF_DISCLOSURE_KEYWORDS:
            if word in text:
                all_disclosure_hits.append(word)

    disclosure_count = len(all_disclosure_hits)
    if disclosure_count == 0:
        disclosure_score = 0
    elif disclosure_count <= 2:
        disclosure_score = 10
    elif disclosure_count <= 4:
        disclosure_score = 20
    else:
        disclosure_score = 30

    total = length_score + emotion_score + disclosure_score

    if total < 40:
        phase = "Phase 1 (破冰陪聊)"
    elif total < 70:
        phase = "Phase 1->2 過渡期"
    else:
        phase = "Phase 2 (深度諮商)"

    return {
        "total_score": total,
        "phase": phase,
        "length_score": length_score,
        "emotion_score": emotion_score,
        "disclosure_score": disclosure_score,
        "emotion_hits": list(set(all_emotion_hits)),
        "disclosure_hits": list(set(all_disclosure_hits)),
    }


# ============================================================
# 根據信任分數，動態生成 Prompt 注入指令
# ============================================================
def build_trust_prompt(trust_result: dict) -> str:
    """
    根據信任評估結果，生成要注入到 LLM System Prompt 的信任閘門指令。
    這段指令會被加在原本 PSYCHOLOGY_PROMPT 之後，控制 AI 能做什麼。
    """
    score = trust_result["total_score"]

    if score < 40:
        return """【信任閘門 - Phase 1：破冰模式（信任分數 < 40）】
你現在的唯一目標是：讓使用者覺得跟你聊天很自在。
嚴格規則：
1. 絕對不准給任何建議或解決方案
2. 絕對不准問太私密的問題（例如：你為什麼難過？你家人呢？）
3. 只能聊輕鬆的日常話題（天氣、吃什麼、看什麼電視）
4. 回覆要簡短溫暖，2~3 句話就好
5. 語氣要像鄰居在聊天，不要像醫生在問診"""

    elif score < 70:
        return """【信任閘門 - 過渡期（信任分數 40~70）】
使用者已經開始願意跟你多聊一些了。你可以開始試探性地問更深的問題。
規則：
1. 可以順著他的話題，問「後來呢？」「那你當時怎麼想的？」
2. 可以表達同理心：「聽起來你那時候一定很不容易」
3. 還是不准主動給建議
4. 如果他提到家人或過去，可以輕輕追問，但不要逼他
5. 回覆 2~4 句話，保持溫暖"""

    else:
        return """【信任閘門 - Phase 2：深度諮商模式（信任分數 ≥ 70）】
使用者已經非常信任你，你現在可以進行深度的心理諮商對話。
諮商策略：
1. 可以深度同理他的感受，幫助他覺察更深層的情緒
2. 可以給出具體的行動建議
3. 回覆 3~5 句話，語氣溫暖但有深度
4. 結尾可以用開放式問句引導他繼續說"""
