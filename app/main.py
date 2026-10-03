import streamlit as st
import sys
import os

# Ensure the root directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.symptom_agent import analyze_symptoms
from agents.clarification_agent import get_clarification_questions
from agents.Safety.triage_agent import assess_safety
from rag.rag_pipeline import medical_rag

# Page configuration
st.set_page_config(
    page_title="Agentic AI Medical Diagnosis System",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Agentic AI Medical Diagnosis & Triage Assistant")
st.caption("A Multi-Agent Architecture grounded in verified World Health Organization (WHO) Knowledge")

# Initialize session state variables
if "stage" not in st.session_state:
    st.session_state.stage = "input"
if "symptoms_input" not in st.session_state:
    st.session_state.symptoms_input = ""
if "symptom_result" not in st.session_state:
    st.session_state.symptom_result = None
if "clarification_result" not in st.session_state:
    st.session_state.clarification_result = None
if "triage_result" not in st.session_state:
    st.session_state.triage_result = None
if "rag_result" not in st.session_state:
    st.session_state.rag_result = None

# Sidebar architecture visualizer
with st.sidebar:
    st.header("🤖 Multi-Agent Architecture")
    st.markdown("""
    1. **Symptom Agent**: Extracts symptoms, duration, and severity.
    2. **Clarification Agent**: Identifies missing clinical variables.
    3. **Triage / Safety Agent**: Checks for critical emergency red flags.
    4. **Specialty Router & RAG**: Grounds assessment in WHO vector DB.
    """)
    st.markdown("---")
    if st.button("🔄 Start New Consultation"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ----------------- STAGE 1: INITIAL SYMPTOMS -----------------
if st.session_state.stage == "input":
    st.subheader("Step 1: Describe Your Symptoms")
    user_input = st.text_area(
        "Enter your current symptoms, duration, and any details:",
        placeholder="e.g., I have high fever, chills, sweating, headache, and body aches for the past 3 days.",
        height=120,
    )
    
    if st.button("Analyze Symptoms", type="primary"):
        if not user_input.strip():
            st.warning("Please enter your symptoms before continuing.")
        else:
            with st.spinner("Analyzing symptoms and drafting clinical clarifications..."):
                st.session_state.symptoms_input = user_input
                st.session_state.symptom_result = analyze_symptoms(user_input)
                st.session_state.clarification_result = get_clarification_questions(st.session_state.symptom_result)
                st.session_state.stage = "clarification"
                st.rerun()

# ----------------- STAGE 2: CLARIFICATION QUESTIONS -----------------
elif st.session_state.stage == "clarification":
    st.info(f"**Reported Symptoms:** {st.session_state.symptoms_input}")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🔍 Symptom Extraction")
        st.write(st.session_state.symptom_result)
        
    with col2:
        st.subheader("❓ Clarification Questions")
        st.markdown(st.session_state.clarification_result)
        
    st.markdown("---")
    st.subheader("Provide Answers to Questions")
    clarification_input = st.text_area(
        "Answer any of the questions above (or click 'Skip' if unsure):",
        placeholder="e.g., Severity is moderate (6/10), no recent travel, no nausea.",
        height=100,
    )
    
    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        submit_answers = st.button("Submit Details", type="primary")
    with col_btn2:
        skip_answers = st.button("Skip / No More Info")
        
    if submit_answers or skip_answers:
        with st.spinner("Assessing safety triage and querying WHO knowledge base..."):
            answers = clarification_input if submit_answers and clarification_input.strip() else "No additional information provided."
            combined_info = f"""
Original User Symptoms:
{st.session_state.symptoms_input}

Symptom Analysis:
{st.session_state.symptom_result}

Clarification Answers:
{answers}
"""
            st.session_state.triage_result = assess_safety(combined_info)
            st.session_state.rag_result = medical_rag(combined_info)
            st.session_state.stage = "results"
            st.rerun()

# ----------------- STAGE 3: RESULTS & RAG -----------------
elif st.session_state.stage == "results":
    # 1. Triage Banner
    triage_str = str(st.session_state.triage_result).upper()
    
    # Check specifically for high triage classification levels
    is_high_risk = any(tag in triage_str for tag in [
        "TRIAGE LEVEL:\n- HIGH", 
        "TRIAGE LEVEL: HIGH", 
        "LEVEL: HIGH",
        "EMERGENCY CARE REQUIRED",
        "RED FLAGS: IDENTIFIED"
    ])
    
    if is_high_risk:
        st.error("🚨 **SAFETY TRIAGE ALERT: HIGH RISK / RED FLAGS IDENTIFIED**")
    else:
        st.success("✅ **SAFETY TRIAGE: LOW / NON-EMERGENCY**")
        
    # 2. Main RAG Assessment
    st.subheader("📋 Medical Assessment (WHO Grounded)")
    st.markdown(st.session_state.rag_result)
    
    st.warning("⚠️ **Medical Disclaimer:** This system provides educational information only and does not provide a confirmed medical diagnosis or individualized treatment recommendation.")
    
    if st.button("Start Another Assessment"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()