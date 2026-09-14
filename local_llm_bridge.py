import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import streamlit as st
import os

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = os.path.join(os.path.dirname(__file__), "ah_guang_v4_lora")

# 這裡使用包含游擊式問法與情感反映的新版專業 System Prompt
SYSTEM_PROMPT = """你是阿光，一個溫暖、敏銳且像朋友般的心理陪伴者，具備專業的心理諮商技術。
對話最高指導原則：
1. 【情感反映】：在提問之前，必須先用一句話溫柔地接住並反映使用者的負面情緒（如：「聽起來那種感覺真的很挫折對吧」）。
2. 【游擊式問法】：不要長篇大論的說教或給建議。每次回應的最後，順著話題丟出一個沒有壓力的探索性短句，問完就把主導權還給對方。
3. 【正向肯定】：無條件接納對方的感受是合理的。
請使用台灣繁體中文的口語習慣。請自然地回應，並盡量簡短、口語化。"""

@st.cache_resource(show_spinner="🧠 正在載入本地阿光大腦 (首次約需 30 秒)...")
def load_local_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
    return model, tokenizer

class LocalResponse:
    def __init__(self, text):
        self.text = text

def generate_local_response(user_message: str, history: list) -> LocalResponse:
    model, tokenizer = load_local_model()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for msg in history:
        # Streamlit 的 "model" 角色需轉為 "assistant"
        role = "assistant" if msg["role"] == "model" else msg["role"]
        
        # 確保角色是 ChatML 支援的
        if role not in ["user", "assistant", "system"]:
            role = "user" if role == "user" else "assistant"
            
        content = msg.get("content", "")
        if content:
            messages.append({"role": role, "content": content})
            
    messages.append({"role": "user", "content": user_message})
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=250, 
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            do_sample=True, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    response_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    return LocalResponse(response_text.strip())
