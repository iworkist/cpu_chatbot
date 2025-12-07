# 참고: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps

from openai import OpenAI
import streamlit as st
import base64
import os
import time
import json
from collections import Counter


SYSTEM_PROMPT = """
너는 사용자의 가장 친한 친구처럼 대화하지만, 정보는 정확하고 똑똑하게 제공하는 AI 친구 이름은 "히루" 다.

## 🧠 역할
- 사용자의 질문, 고민, 작업 요청, 계획 등을 이해하고
  "실제로 도움이 되는 답"을 빠르고 쉽게 제공한다.
- 말투는 친근하고 편안하지만, 답변 내용은 논리적이고 실용적이어야 한다.

## 💬 말투 스타일
- 딱딱하지 않게! 그러나 가볍게 넘어가지도 않기.
- 존댓말과 반말은 상황에 따라 자연스럽게, 사용자 의도를 따라간다.
- 이모지 사용은 가능하지만 과하게 쓰지 않는다.

예시 표현:
- "오케이 이해했어 👍"
- "그거 이렇게 하면 더 쉬울걸?"
- "잠깐만, 내가 정리해줄게."

## 📌 답변 구조
모든 답변은 다음 형식을 따른다:

1. **빠른 핵심 답변 (한두 줄)**  
   → "바로 요점 먼저!"

2. **구체적 해결 방법 또는 결과물**  
   → 리스트, 단계, 예시, 코드 등 필요 형태로 제공.

3. **선택지 또는 다음 행동 추천**  
   → "이대로 갈래?" / "아니면 이런 방향도 가능해."

## 🎯 규칙
- 사용자의 질문이 모호하면 1~2문장으로 부드럽게 확인 질문.
- 너무 전문적이라면 쉽게 설명 + 원하면 더 디테일 제공.
- 사용자 시간을 절약하는 방향 우선.

## 🚫 제한
- 무책임한 추측 금지. 모르면 솔직하게 말한 뒤 대안을 제시.
- 장황한 설명, 쓸모없는 역사 배경 설명은 금지.

---

이제부터 너의 목표는:
👉 "사용자가 무엇을 원하든 걱정 없이 편하게 물어보고, 빠르게 해결 가능한 느낌을 주는 똑똑한 친구"  

사용자가 메시지를 보내면:
- 먼저 "내가 이해한 내용"을 짧게 확인하고
- 바로 자연스럽게 대답을 시작한다.


"""

st.markdown("""
<style>
.chat-wrapper {
    display: flex;
    width: 100%;
    margin: 6px 0;
}

.user-bubble {
    background-color: #a0d8ff; 
    color: #000;
    padding: 12px 18px;
    border-radius: 20px;
    border-bottom-right-radius: 0;
    max-width: 75%;
    margin-left: auto;  
    position: relative;
    font-size: 16px;
    line-height: 1.45;
    word-wrap: break-word;
}

.bot-bubble {
    background-color: #f0f0f0; 
    color: #000;
    padding: 12px 18px;
    border-radius: 20px;
    border-bottom-left-radius: 0;
    max-width: 75%;
    margin-right: auto; 
    position: relative;
    font-size: 16px;
    line-height: 1.45;
    word-wrap: break-word;
    border: 1px solid #dcdcdc;
}

.icon-ai {
    font-size: 20px;
    margin-right: 6px;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)



# Cerebras API를 사용하여 OpenAI API 클라이언트 초기화
client = OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=st.secrets["CEREBRAS_API_KEY"]
)

# Cerebras 모델 사용
# https://inference-docs.cerebras.ai/models/overview
# "qwen-3-32b"
# "qwen-3-235b-a22b-instruct-2507",
# "qwen-3-coder-480b"
# "llama-4-scout-17b-16e-instruct"
# "qwen-3-235b-a22b-thinking-2507"
# "llama-3.3-70b"
# "llama3.1-8b"
# "gpt-oss-120b"
llm_model = "gpt-oss-120b"  
if "llm_model" not in st.session_state:
    st.session_state["llm_model"] = llm_model


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]
if "saved_chats" not in st.session_state:
    st.session_state["saved_chats"] = {}

st.title("하루☁️")


with st.sidebar:
    st.header("💾 대화 관리")
    new_chat_name = st.text_input("대화 이름")

    if st.button("대화 저장") and new_chat_name.strip():
        st.session_state.saved_chats[new_chat_name.strip()] = st.session_state.messages.copy()
        st.success(f"✅ '{new_chat_name.strip()}' 저장 완료")

    if st.button("새 채팅"):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


    st.subheader("💬 저장된 대화")
    for chat_name in st.session_state.saved_chats.keys():
        if st.button(chat_name):
            st.session_state.messages = st.session_state.saved_chats[chat_name].copy()

def render_message(role, content):
    bubble = "user-bubble" if role == "user" else "bot-bubble"
    icon = "☁️ " if role == "assistant" else ""
    if isinstance(content, list):
        for item in content:
            if item.get("type") == "text":
                st.markdown(f"""
                <div class="chat-wrapper">
                    <div class="{bubble}">{icon}{item['text']}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-wrapper">
            <div class="{bubble}">{icon}{content}</div>
        </div>
        """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] != "system":
        render_message(msg["role"], msg["content"])

def generate_recommendation(user_chat_log):
    keywords = ["AI","코딩","파이썬","연애","노래","운동","일정","계획","공부","디자인"]
    text = " ".join([m["content"][0]["text"] if isinstance(m["content"], list) else m["content"]
                     for m in user_chat_log if m["role"]=="user"])
    score = Counter({k:text.count(k) for k in keywords})
    if not score or max(score.values())==0:
        return "👌 궁금한 거 아무거나 물어봐도 돼!"
    topic = score.most_common(1)[0][0]
    return f"🤔 혹시 '{topic}' 관련해서 더 알고 싶어?"

prompt = st.chat_input("궁금한 게 있으면 물어봐 !", key="chat_input")

if prompt:
    user_content = [{"type": "text", "text": prompt}]
    st.session_state.messages.append({"role":"user","content":user_content})
    render_message("user", user_content)

    placeholder = st.empty()
    ai_text = ""

    stream = client.chat.completions.create(
        model=st.session_state["llm_model"],
        messages=[{"role":"system","content":SYSTEM_PROMPT}] + st.session_state.messages,
        stream=True,
        temperature=0.6
    )

    for chunk in stream:
        delta = getattr(chunk.choices[0], "delta", None)
        if delta and hasattr(delta, "content") and delta.content:
            ai_text += delta.content
            placeholder.markdown(f"""
            <div class="chat-wrapper">
                <div class="bot-bubble">☁️ {ai_text}▋</div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.03)  

    placeholder.markdown(f"""
    <div class="chat-wrapper">
        <div class="bot-bubble">☁️ {ai_text}</div>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.messages.append({"role":"assistant","content":ai_text})

    reco = generate_recommendation(st.session_state.messages)
    st.session_state.messages.append({"role":"assistant","content":reco})
    render_message("assistant", reco)

# 자동 실행 지원
if __name__ == "__main__":
    import subprocess
    import sys

    if not os.environ.get("STREAMLIT_RUNNING"):
        os.environ["STREAMLIT_RUNNING"] = "1"
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])

# python -m streamlit run main.py
# streamlit run main.py