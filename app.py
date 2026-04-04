import streamlit as st
import pandas as pd
import json

# Module imports
from profiler import generate_data_catalog, catalog_to_json
from llm_client import (
    configure_gemini, generate_with_gemini, generate_with_ollama, 
    build_prompt, build_classification_prompt, parse_classification_response,
    build_xai_prompt
)
from preprocessor import preprocess_dataframe
from templates import render_template

st.set_page_config(page_title="DropML: AI Data Analysis", page_icon="⚡", layout="wide")

st.title("⚡ DropML")
st.markdown("**Drop a CSV. Ask a Question. Get Answers.** — AI analyzes your data, generates code, and explains the results.")

# ==========================================
# 1. Sidebar (Settings)
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "Select AI Model:",
        ("Gemini 1.5 Pro (Cloud)", "Llama 3 (Local API)")
    )
    
    api_key = ""
    if "Gemini" in model_choice:
        api_key = st.text_input("Enter Gemini API Key", type="password")
        if api_key:
            configure_gemini(api_key)
            
    st.markdown("---")
    st.markdown("💡 **Tip:** To use Llama 3, make sure the Ollama server is running locally. (`http://localhost:11434`)")

# ==========================================
# 2. Main Area (Data Upload & Profiling)
# ==========================================
st.markdown("### 📂 1. Upload Data")
uploaded_file = st.file_uploader("Upload a file to analyze (CSV, Excel, JSON)", type=["csv", "xlsx", "xls", "json"])

if uploaded_file is not None:
    # Read data
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else: # xlsx, xls
            df = pd.read_excel(uploaded_file)
            
        # Pre-process data globally
        df = preprocess_dataframe(df)
            
        st.write(f"✅ **Preview (Top 5 Rows)** - `{uploaded_file.name}`")
        st.dataframe(df.head())
        
        # Extract data catalog
        with st.spinner("Scanning data and generating catalog..."):
            catalog = generate_data_catalog(df)
            catalog_json_str = catalog_to_json(catalog)
            
        with st.expander("📊 View Extracted Data Catalog (Schema & Stats)"):
            st.json(catalog)
            
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # ==========================================
    # 3. User Goal Input & Hybrid Execution Engine
    # ==========================================
    st.markdown("---")
    st.markdown("### 🎯 2. Set Analysis Goal & Run")
    user_goal = st.text_area("What analysis or prediction model would you like to build?", 
                             placeholder="Example: Show the top 5 countries by employee count as a pie chart.", height=100)

    if st.button("🚀 Generate & Execute Analysis Code", type="primary"):
        if not user_goal.strip():
            st.warning("Please enter an analysis goal.")
            st.stop()
            
        if "Gemini" in model_choice and not api_key:
            st.warning("Please enter your Gemini API Key in the sidebar.")
            st.stop()
            
        from executor import run_generated_code, classify_error, build_self_healing_prompt
        
        st.markdown("---")
        st.markdown("### 💡 AI Execution Results")
        
        # ============================================
        # STEP 1: Hybrid Routing — Template Classification
        # ============================================
        template_code = None
        template_id = None
        params = {}
        
        with st.spinner("🧠 Analyzing your request and determining the optimal execution path..."):
            goal_lower = user_goal.lower()
            
            # Attempt to extract actual column names from user text
            _target_col = None
            for col in df.columns:
                if col.lower() in goal_lower:
                    _target_col = col
            if not _target_col:
                _target_col = df.columns[-1]  # Default to last column as target
            
            # ---- [Tier 1] Keyword Priority Matching: Route obvious patterns instantly ----
            if any(kw in goal_lower for kw in ["중요한 변수", "변수 중요도", "feature importance", "important variable", "importance"]):
                template_id = "feature_importance"
                params = {"target_col": _target_col, "n": 5}
            elif any(kw in goal_lower for kw in ["예측", "분류 모델", "classification", "predict", "classifier"]):
                template_id = "classification_model"
                params = {"target_col": _target_col}
            elif any(kw in goal_lower for kw in ["상관관계", "correlation", "히트맵", "heatmap"]):
                template_id = "correlation_heatmap"
                params = {}
            
            # ---- [Tier 2] LLM Classification: AI decides for unmatched requests ----
            if not template_id:
                classification_prompt = build_classification_prompt(catalog_json_str, user_goal)
                
                if "Gemini" in model_choice:
                    classification_response = generate_with_gemini(classification_prompt)
                else:
                    classification_response = generate_with_ollama(classification_prompt, model_name="llama3")
                
                parsed = parse_classification_response(classification_response)
                
                if parsed and parsed.get("template_id") and parsed["template_id"] != "none":
                    template_id = parsed["template_id"]
                    params = parsed.get("params", {})
            
            # Render template code if a template was matched
            if template_id and template_id != "none":
                template_code = render_template(template_id, params, df_columns=df.columns.tolist())
                
                if template_code:
                    st.info(f"🎯 **Template Mode**: `{template_id}` pattern detected. Using pre-validated code.")
        
        # ============================================
        # STEP 2A: Template Code Execution (Safe Path)
        # ============================================
        # Variables to store successful code for XAI explanation
        xai_success_code = None
        xai_output = ""
        
        if template_code:
            with st.spinner("Executing template code..."):
                success, final_code, error_trace = run_generated_code(template_code, df)
                
            if success:
                st.success("Code executed successfully! 🎉 (Results are displayed above.)")
                with st.expander("💻 (Click to expand) Executed Template Code"):
                    st.code(final_code, language="python")
                xai_success_code = final_code
                xai_output = error_trace if error_trace else "(Chart/visualization output complete)"
            else:
                # Template failed → Fallback to free-form code generation
                st.warning("Template code execution failed. Switching to free-form code generation...")
                template_code = None  # Trigger fallback
        
        # ============================================
        # STEP 2B: Free-Form Code Generation + Self-Healing
        # ============================================
        if not template_code:
            st.info("🔮 **Free-Form Mode**: No matching template found. AI is writing custom code.")
            
            MAX_RETRIES = 3
            current_prompt = build_prompt(catalog_json_str, user_goal)
            
            for attempt in range(MAX_RETRIES):
                with st.spinner(f"[{model_choice}] Generating & executing code... (Attempt {attempt+1}/{MAX_RETRIES})"):
                    # 1. Generate code
                    if "Gemini" in model_choice:
                        response_text = generate_with_gemini(current_prompt)
                    else:
                        response_text = generate_with_ollama(current_prompt, model_name="llama3")
                    
                    # 2. Execute code (through 3-layer defense)
                    success, final_code, error_trace = run_generated_code(response_text, df)
                    
                    # 3. Branch based on success/failure
                    if success:
                        st.success("Code executed successfully! 🎉 (Results are displayed above.)")
                        with st.expander("💻 (Click to expand) AI-Generated Code"):
                            st.code(final_code, language="python")
                        xai_success_code = final_code
                        xai_output = error_trace if error_trace else "(Chart/visualization output complete)"
                        break
                    else:
                        # Layer 3: Smart Self-Healing
                        error_info = classify_error(error_trace)
                        
                        st.warning(f"Runtime error detected. AI is diagnosing and attempting to self-heal... (Attempt {attempt+1}/{MAX_RETRIES}) [Error Type: {error_info['type']}]")
                        with st.expander("View Error Details", expanded=False):
                            st.code(error_trace)
                            st.info(f"🔍 **Auto-Diagnosis:** {error_info['hint']}")
                            
                        current_prompt = build_self_healing_prompt(final_code, error_trace, error_info)
            else:
                st.error(f"Maximum retry limit ({MAX_RETRIES} attempts) exceeded. Unable to fully resolve the error.")
        
        # ============================================
        # STEP 3: XAI — Explainable AI Analysis
        # ============================================
        if xai_success_code:
            with st.spinner("🧠 AI is interpreting the analysis results..."):
                xai_prompt = build_xai_prompt(
                    catalog_json_str, user_goal, xai_success_code, xai_output
                )
                if "Gemini" in model_choice:
                    xai_response = generate_with_gemini(xai_prompt)
                else:
                    xai_response = generate_with_ollama(xai_prompt, model_name="llama3")
                
                st.markdown("---")
                st.markdown("### 🧠 AI Analysis Interpretation (XAI)")
                st.markdown(xai_response)
