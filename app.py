import os
import sys
import time
import streamlit as st

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.core.openai_provider import OpenAIProvider
from src.core.gemini_provider import GeminiProvider
from src.core.local_provider import LocalProvider
from src.agent.agent import ReActAgent
from chatbot import BaselineChatbot
from src.telemetry.metrics import tracker
from test_agent import TOOLS_SPEC

# Page configuration
st.set_page_config(
    page_title="AI Nutrition Lab - Agent vs Chatbot",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #10B981, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    
    .card {
        border-radius: 12px;
        padding: 1.5rem;
        background-color: #FFFFFF;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
    
    .step-card {
        background-color: #F9FAFB;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        border-radius: 4px 8px 8px 4px;
        margin-bottom: 0.8rem;
    }
    
    .observation-card {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 0.8rem;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    .metrics-header {
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="main-title">🥗 AI Nutrition Lab Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Compare ReAct Agent reasoning with tool access vs Baseline Chatbot</div>', unsafe_allow_html=True)

# Sidebar settings
st.sidebar.markdown("### ⚙️ System Settings")

provider_name = st.sidebar.selectbox(
    "Select LLM Provider",
    ["Google Gemini", "OpenAI", "Local (Phi-3)"],
    index=0
)

# Set model options based on provider
if provider_name == "Google Gemini":
    model_name = st.sidebar.selectbox("Select Model", ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"], index=0)
elif provider_name == "OpenAI":
    model_name = st.sidebar.selectbox("Select Model", ["gpt-4o", "gpt-4o-mini"], index=0)
else:
    model_name = st.sidebar.text_input("Local Model Path", os.getenv("LOCAL_MODEL_PATH", "./models/Phi-3-mini-4k-instruct-q4.gguf"))

mode = st.sidebar.radio(
    "Running Mode",
    ["ReAct Agent", "Baseline Chatbot"],
    index=0
)

max_steps = st.sidebar.slider("Max ReAct Steps", 3, 12, 8)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💳 Cost & Telemetry Tracker")
if st.sidebar.button("Clear Session Metrics"):
    tracker.session_metrics = []
    st.sidebar.success("Metrics reset!")

# Calculate and display accumulated metrics in sidebar
total_cost = sum(m.get("cost_usd", 0.0) for m in tracker.session_metrics)
total_tokens = sum(m.get("total_tokens", 0) for m in tracker.session_metrics)
total_requests = len(tracker.session_metrics)

st.sidebar.metric("Accumulated Cost", f"${total_cost:.5f}")
st.sidebar.metric("Total API Requests", f"{total_requests}")
st.sidebar.metric("Total Tokens Transferred", f"{total_tokens:,}")

# Initialize API Provider wrapper
@st.cache_resource
def get_llm_provider(p_name, m_name):
    if p_name == "Google Gemini":
        return GeminiProvider(model_name=m_name, api_key=os.getenv("GEMINI_API_KEY"))
    elif p_name == "OpenAI":
        return OpenAIProvider(model_name=m_name, api_key=os.getenv("OPENAI_API_KEY"))
    else:
        return LocalProvider(model_path=m_name)

# Quick Presets
st.markdown("### 📋 Quick Run Preset Scenarios")
col1, col2, col3 = st.columns(3)

preset_query = ""

with col1:
    if st.button("🥩 Scenario 1: Nguyễn Văn A (Tăng Cơ)"):
        preset_query = "Tôi là Nguyễn Văn A (user_1). Hãy tính nhu cầu dinh dưỡng của tôi và thiết kế cho tôi một thực đơn ăn uống trong ngày để tăng cơ."
        st.session_state.preset = preset_query

with col2:
    if st.button("🥗 Scenario 2: Trần Thị B (Giảm Cân + Dị Ứng)"):
        preset_query = "Tôi là Trần Thị B (user_2). Thiết kế cho tôi một thực đơn ăn uống giảm cân và tính toán calo chi tiết. Lưu ý tôi bị dị ứng bún riêu cua."
        st.session_state.preset = preset_query

with col3:
    if st.button("🍜 Scenario 3: Tra cứu & Đánh giá Dinh Dưỡng"):
        preset_query = "Hôm nay tôi đã ăn: 1 tô Phở bò, 1 đĩa Cơm tấm sườn nướng, và 1 bát Chè chuối. Hãy tra cứu calo và đánh giá dinh dưỡng giúp tôi."
        st.session_state.preset = preset_query

# Input choice
st.markdown("### 💬 Input Information")
input_type = st.radio(
    "Chọn phương thức nhập thông tin:",
    ["Form Nhập Thông Tin Cá Nhân (Structured Form)", "Nhập Câu Hỏi Tự Do (Free-form Query)"],
    index=0
)

user_input = ""

if input_type == "Form Nhập Thông Tin Cá Nhân (Structured Form)":
    with st.container():
        st.markdown('<div style="background-color: #f0f2f6; padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem;">', unsafe_allow_html=True)
        form_col1, form_col2, form_col3 = st.columns(3)
        
        with form_col1:
            form_user_id = st.text_input("Mã người dùng hoặc Tên (User ID/Name)", value="user_1")
            form_age = st.number_input("Tuổi (Age)", min_value=1, max_value=120, value=24, step=1)
            form_gender = st.selectbox("Giới tính (Gender)", ["Nam", "Nữ"], index=0)
            
        with form_col2:
            form_height = st.number_input("Chiều cao (Height in cm)", min_value=50.0, max_value=250.0, value=175.0, step=0.5)
            form_weight = st.number_input("Cân nặng (Weight in kg)", min_value=10.0, max_value=300.0, value=68.0, step=0.5)
            
        with form_col3:
            form_activity = st.selectbox(
                "Mức độ hoạt động (Activity Level)", 
                [
                    "sedentary (Ít vận động)", 
                    "light (Vận động nhẹ)", 
                    "moderate (Vận động vừa)", 
                    "active (Vận động nhiều)", 
                    "very_active (Vận động cực nhiều)"
                ], 
                index=2
            )
            form_goal = st.selectbox(
                "Mục tiêu dinh dưỡng (Goal)", 
                [
                    "build_muscle (Tăng cơ)", 
                    "lose_weight (Giảm cân)", 
                    "gain_weight (Tăng cân)", 
                    "maintain (Duy trì cân nặng)"
                ], 
                index=0
            )
            
        form_exclude = st.text_input("Dị ứng / Thực phẩm cần tránh (Ví dụ: bún riêu cua, đậu phộng)", value="")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Compile structured prompt
        act_val = form_activity.split(" ")[0]
        goal_val = form_goal.split(" ")[0]
        exclude_part = f", tôi bị dị ứng hoặc muốn tránh: {form_exclude}" if form_exclude.strip() else ""
        
        user_input = (
            f"Tôi là {form_user_id}. Tuổi: {form_age}, Giới tính: {form_gender}, "
            f"Chiều cao: {form_height}cm, Cân nặng: {form_weight}kg. "
            f"Mức độ hoạt động: {act_val}, Mục tiêu: {goal_val}{exclude_part}. "
            f"Hãy tính BMI, BMR, TDEE, lượng macro và lập thực đơn ăn uống trong ngày cho tôi."
        )
        
        st.info(f"**Generated Query:** {user_input}")

else:
    user_input = st.text_area(
        "Enter your nutritional query:",
        value=st.session_state.get("preset", ""),
        placeholder="Nhập thông tin cân nặng, chiều cao, mục tiêu hoặc ghi nhật ký ăn uống của bạn ở đây...",
        key="query_input"
    )

run_clicked = st.button("🔥 Run Analysis", type="primary")

if run_clicked and user_input:
    # Set up LLM
    try:
        llm = get_llm_provider(provider_name, model_name)
    except Exception as e:
        st.error(f"Failed to initialize LLM provider: {e}")
        st.stop()

    st.markdown("---")
    
    # Visual columns for Live Execution vs Metrics
    exec_col, metrics_col = st.columns([3, 1])
    
    with exec_col:
        st.markdown(f"#### ⚙️ Running: **{mode}** using model `{model_name}`")
        
        # Reset tracker for current execution
        tracker.session_metrics = []
        start_time = time.time()
        
        with st.spinner("Processing request..."):
            if mode == "ReAct Agent":
                # Instantiating ReAct Agent
                agent = ReActAgent(llm=llm, tools=TOOLS_SPEC, max_steps=max_steps)
                
                # Executing agent run inside a placeholder to capture logs/steps if needed
                response = agent.run(user_input)
                
                st.markdown("### 🧩 Agent Reasoning Steps:")
                for step in agent.steps_history:
                    with st.expander(f"Step {step['step']}: {step['action'] or 'Final Synthesis'}", expanded=True):
                        st.markdown(f"**Thought:** {step['thought']}")
                        if step['action']:
                            st.markdown(f"**Action Called:** `{step['action']}`")
                        if step['observation']:
                            st.markdown("**Observation (Tool Output):**")
                            st.markdown(f'<div class="observation-card">{step["observation"]}</div>', unsafe_allow_html=True)
                
                st.markdown("### 🎯 Final Answer")
                st.info(response)
                
            else:
                # Instantiating Chatbot
                chatbot = BaselineChatbot()
                response = chatbot.generate_response(user_input)
                st.markdown("### 🎯 Final Answer")
                st.info(response)
                
        execution_time = (time.time() - start_time) * 1000  # in ms
        
    with metrics_col:
        st.markdown('<div class="metrics-header">📊 Execution Telemetry</div>', unsafe_allow_html=True)
        
        # Compute run metrics
        run_tokens = sum(m.get("total_tokens", 0) for m in tracker.session_metrics)
        run_cost = sum(m.get("cost_usd", 0.0) for m in tracker.session_metrics)
        run_steps = len(tracker.session_metrics)
        
        st.metric("Total Latency", f"{execution_time/1000:.2f} s")
        st.metric("Steps Executed", f"{run_steps}")
        st.metric("Tokens Transferred", f"{run_tokens:,}")
        st.metric("Estimated Cost", f"${run_cost:.5f}")
        
        st.markdown("---")
        st.markdown("💡 **Tip**: Notice how the **ReAct Agent** makes multiple steps and calls actual Python calculation functions to ensure BMR/TDEE accuracy, while the **Baseline Chatbot** generates the answer directly in 1 step.")
