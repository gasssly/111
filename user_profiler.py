"""
======================================================
 使用者畫像提取器 (User Profiler)
======================================================
 純 Library 版本：供 main.py import 使用。
 從對話中無痕提取使用者的個性特徵。
 不問問卷，純粹從「他怎麼說話」和「他說了什麼」來推斷。

 提取的維度：
   1. 因應型態 (Coping Style)：情緒導向 vs 問題導向
   2. 社交傾向 (Social Tendency)：內向 vs 外向
   3. 表達程度 (Expression Level)：沉默寡言 vs 願意傾訴
   4. 核心議題 (Core Concerns)：反覆出現的主題

 理論基礎：
   - Lazarus & Folkman (1984) 壓力因應理論
======================================================
"""


# ==================================================
# 因應型態關鍵字
# ==================================================
EMOTION_FOCUSED_WORDS = {
    "難過", "傷心", "心痛", "想哭", "好累", "無力", "無助",
    "寂寞", "孤單", "空虛", "害怕", "不安", "焦慮",
    "委屈", "心酸", "崩潰", "撐不住", "受傷",
    "我覺得", "我感覺", "我心裡", "我好像",
    "不知道為什麼", "就是很", "莫名其妙",
}

PROBLEM_FOCUSED_WORDS = {
    "怎麼辦", "該怎麼", "有沒有辦法", "怎麼做",
    "我想要", "我打算", "我決定", "我可以",
    "有什麼方法", "怎麼改善", "怎麼解決",
    "無聊", "不知道可以幹嘛", "想找事做",
    "應該", "可以嗎", "好不好",
}

# ==================================================
# 社交傾向關鍵字
# ==================================================
INTROVERT_WORDS = {
    "一個人", "自己", "安靜", "不想出門", "待在家",
    "不太想", "懶得", "不想見人", "好煩", "不想動",
    "房間", "獨處", "靜靜的",
}

EXTROVERT_WORDS = {
    "找人", "聊天", "出去", "朋友", "大家",
    "一起", "熱鬧", "無聊想找人", "約", "陪我",
    "想出門", "散步", "走走",
}

# ==================================================
# 核心議題關鍵字群組
# ==================================================
CONCERN_CATEGORIES = {
    "孤獨": {"孤單", "寂寞", "一個人", "沒人", "陪", "獨居", "空虛"},
    "家庭": {"爸", "媽", "兒子", "女兒", "家人", "家裡", "老婆", "老公", "孫"},
    "健康": {"痛", "不舒服", "看醫生", "吃藥", "失眠", "睡不著", "住院", "生病", "累"},
    "失落": {"走了", "不在了", "過世", "離開", "以前", "懷念", "想念", "回不去"},
    "經濟": {"錢", "薪水", "工作", "沒錢", "房租", "貸款", "開銷"},
    "人際": {"吵架", "冷戰", "不理我", "討厭", "被罵", "不信任", "背叛"},
}


def analyze_profile(user_turns: list) -> dict:
    """
    輸入使用者的所有歷史發言，輸出個性畫像。
    """
    if not user_turns:
        return {
            "coping_style": "unknown",
            "coping_detail": {"emotion": 0, "problem": 0},
            "social_tendency": "unknown",
            "social_detail": {"introvert": 0, "extrovert": 0},
            "expression_level": "silent",
            "core_concerns": [],
            "total_turns": 0,
        }

    all_text = " ".join(user_turns)

    # --- 因應型態 ---
    emotion_count = sum(1 for w in EMOTION_FOCUSED_WORDS if w in all_text)
    problem_count = sum(1 for w in PROBLEM_FOCUSED_WORDS if w in all_text)

    if emotion_count == 0 and problem_count == 0:
        coping = "unknown"
    elif emotion_count >= problem_count:
        coping = "emotion-focused"
    else:
        coping = "problem-focused"

    # --- 社交傾向 ---
    intro_count = sum(1 for w in INTROVERT_WORDS if w in all_text)
    extro_count = sum(1 for w in EXTROVERT_WORDS if w in all_text)

    if intro_count == 0 and extro_count == 0:
        social = "unknown"
    elif intro_count >= extro_count:
        social = "introvert"
    else:
        social = "extrovert"

    # --- 表達程度 ---
    avg_len = sum(len(t) for t in user_turns) / len(user_turns)
    if avg_len <= 5:
        expression = "silent"
    elif avg_len <= 20:
        expression = "reserved"
    elif avg_len <= 50:
        expression = "moderate"
    else:
        expression = "expressive"

    # --- 核心議題 ---
    concern_scores = {}
    for category, keywords in CONCERN_CATEGORIES.items():
        score = sum(1 for w in keywords if w in all_text)
        if score > 0:
            concern_scores[category] = score

    concerns = sorted(concern_scores, key=concern_scores.get, reverse=True)[:3]

    return {
        "coping_style": coping,
        "coping_detail": {"emotion": emotion_count, "problem": problem_count},
        "social_tendency": social,
        "social_detail": {"introvert": intro_count, "extrovert": extro_count},
        "expression_level": expression,
        "core_concerns": concerns,
        "total_turns": len(user_turns),
    }


def build_profile_prompt(profile: dict) -> str:
    """
    根據畫像分析結果，生成要注入到 LLM System Prompt 的個人化指令。
    """
    parts = []

    # 因應型態
    if profile["coping_style"] == "emotion-focused":
        parts.append("使用者屬於「情緒導向」型，需要被傾聽和接住。請使用 EFT 情緒焦點策略，著重同理心與情緒接納，不要急著給建議。")
    elif profile["coping_style"] == "problem-focused":
        parts.append("使用者屬於「問題導向」型，想要解決方案。請使用 SFBT 焦點解決策略，適時提供具體可行的行動建議。")

    # 社交傾向
    if profile["social_tendency"] == "introvert":
        parts.append("使用者偏內向，禁止建議出門社交或找人聊天。給予安靜獨處的空間。")
    elif profile["social_tendency"] == "extrovert":
        parts.append("使用者偏外向，可以建議他出門走走、找朋友聊聊或參加活動。")

    # 表達程度（控制回覆長度）
    if profile["expression_level"] == "silent":
        parts.append("使用者目前非常沉默，回覆請控制在 1~2 句話，不要問太多問題。")
    elif profile["expression_level"] == "reserved":
        parts.append("使用者話不多但願意回應，回覆 2~3 句話即可。")
    elif profile["expression_level"] == "expressive":
        parts.append("使用者很願意傾訴，可以回覆 3~5 句話做更深入的對話。")

    # 核心議題
    if profile["core_concerns"]:
        concerns_str = "、".join(profile["core_concerns"])
        parts.append(f"使用者目前最關注的議題是：{concerns_str}。請圍繞這些主題展開對話。")

    if not parts:
        return ""

    return "【使用者畫像分析結果】\n" + "\n".join(parts)
