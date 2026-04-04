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

# ==========================================
# 🎨 GLOBAL CUSTOM CSS
# ==========================================
st.markdown("""
<style>
/* ── Google Font: Inter ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, .stDeployButton { visibility: hidden; display: none !important; }
div[data-testid="stToolbar"] { visibility: hidden; }

/* ── Page background ── */
.stApp {
    background: linear-gradient(135deg, #0D1117 0%, #0f1923 60%, #0D1117 100%);
    background-attachment: fixed;
}

/* ── Remove default block padding from main area ── */
.block-container {
    padding-top: 1.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* ─────────────────────────────────────────
   HERO
───────────────────────────────────────── */
.hero-section {
    text-align: center;
    padding: 3rem 2rem 2.2rem;
    background: linear-gradient(135deg, rgba(136,189,242,0.07) 0%, rgba(58,75,89,0.04) 100%);
    border-radius: 20px;
    border: 1px solid rgba(136,189,242,0.14);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-section::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(136,189,242,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-logo {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #88BDF2 0%, #BDDDFC 60%, #6A89A7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.hero-tagline {
    font-size: 1.05rem;
    color: #8B949E;
    margin-bottom: 0.3rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(136,189,242,0.08);
    border: 1px solid rgba(136,189,242,0.18);
    border-radius: 20px;
    padding: 5px 16px;
    font-size: 0.78rem;
    color: #88BDF2;
    font-weight: 500;
    margin-top: 0.9rem;
}

/* ─────────────────────────────────────────
   SECTION HEADERS (pure HTML elements only)
───────────────────────────────────────── */
.section-label {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.74rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #88BDF2;
    margin-bottom: 0.4rem;
    opacity: 0.85;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(136,189,242,0.3), transparent);
}
.step-header  { font-size: 1.05rem; font-weight: 700; color: #E6EDF3; margin: 0; }
.step-sub     { font-size: 0.83rem; color: #6E7681; margin: 0.2rem 0 0; }

/* ─────────────────────────────────────────
   GLASS CARD — applied to Streamlit containers
   via CSS class injection trick on st.container
───────────────────────────────────────── */
div[data-testid="stVerticalBlock"] > div.element-container + div.element-container { }

/* CSS class we add via st.markdown + adjacent sibling targeting */
.card-wrap {
    background: rgba(22, 27, 34, 0.72);
    border: 1px solid rgba(136,189,242,0.12);
    border-radius: 16px;
    padding: 1.4rem 1.5rem 1.5rem;
    margin-bottom: 1.2rem;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.card-wrap:hover {
    border-color: rgba(136,189,242,0.22);
    box-shadow: 0 4px 28px rgba(136,189,242,0.07);
}

/* ─────────────────────────────────────────
   FILE UPLOADER — style the widget itself
───────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(136,189,242,0.03) !important;
    border: 2px dashed rgba(136,189,242,0.22) !important;
    border-radius: 14px !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.25s ease !important;
}
[data-testid="stFileUploader"]:hover {
    background: rgba(136,189,242,0.07) !important;
    border-color: rgba(136,189,242,0.42) !important;
    box-shadow: 0 4px 20px rgba(136,189,242,0.08) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* ─────────────────────────────────────────
   TEXT AREA — style the widget itself
───────────────────────────────────────── */
[data-testid="stTextArea"] {
    border-radius: 14px !important;
    overflow: hidden;
}
[data-testid="stTextArea"] > div {
    background: rgba(22, 27, 34, 0.85) !important;
    border: 1.5px solid rgba(136,189,242,0.2) !important;
    border-radius: 14px !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
    padding: 0.25rem !important;
}
[data-testid="stTextArea"] > div:focus-within {
    border-color: rgba(136,189,242,0.5) !important;
    box-shadow: 0 0 0 3px rgba(136,189,242,0.08), 0 4px 24px rgba(136,189,242,0.1) !important;
}
[data-testid="stTextArea"] textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #E6EDF3 !important;
    font-size: 0.93rem !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.65 !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: #404858 !important; }

/* ─────────────────────────────────────────
   BUTTONS
───────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #384959 0%, #6A89A7 55%, #88BDF2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.62rem 1.5rem !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    color: #fff !important;
    transition: all 0.28s ease !important;
    box-shadow: 0 4px 18px rgba(136,189,242,0.22) !important;
    width: 100% !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(136,189,242,0.38) !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }
.stButton > button { border-radius: 10px !important; font-family: 'Inter', sans-serif !important; }

/* ─────────────────────────────────────────
   METRIC GRID
───────────────────────────────────────── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
    gap: 0.7rem;
    margin-top: 0.8rem;
}
.metric-card {
    background: rgba(136,189,242,0.05);
    border: 1px solid rgba(136,189,242,0.12);
    border-radius: 12px;
    padding: 0.9rem 0.6rem;
    text-align: center;
    transition: all 0.22s ease;
}
.metric-card:hover {
    background: rgba(136,189,242,0.1);
    border-color: rgba(136,189,242,0.3);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(136,189,242,0.1);
}
.metric-value { font-size: 1.55rem; font-weight: 700; color: #88BDF2; line-height: 1.2; }
.metric-label { font-size: 0.68rem; color: #6E7681; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 3px; }

/* ─────────────────────────────────────────
   MISC
───────────────────────────────────────── */
.stDataFrame { border-radius: 12px !important; overflow: hidden; }
hr { border-color: rgba(136,189,242,0.1) !important; }

.stTabs [data-baseweb="tab-list"] {
    background: rgba(22,27,34,0.6); border-radius: 12px; padding: 4px; gap: 4px;
    border: 1px solid rgba(136,189,242,0.1);
}
.stTabs [data-baseweb="tab"] { border-radius: 8px !important; color: #6E7681 !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] { background: rgba(136,189,242,0.15) !important; color: #88BDF2 !important; }

.streamlit-expanderHeader {
    background: rgba(136,189,242,0.04) !important; border-radius: 10px !important;
    font-weight: 600 !important; color: #C9D1D9 !important;
}
.stSuccess { border-left: 3px solid #88BDF2 !important; border-radius: 8px !important; }
.stWarning { border-left: 3px solid #f0a500 !important; border-radius: 8px !important; }
.stError   { border-left: 3px solid #f85149 !important; border-radius: 8px !important; }
.stInfo    { border-left: 3px solid rgba(136,189,242,0.5) !important; border-radius: 8px !important; }

[data-testid="stSidebar"] {
    background: rgba(13,17,23,0.97) !important;
    border-right: 1px solid rgba(136,189,242,0.08) !important;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-thumb { background: rgba(136,189,242,0.18); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(136,189,242,0.35); }

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeInUp 0.4s ease both; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 🦸 HERO
# ==========================================
st.markdown("""
<div class="hero-section fade-in">
    <div class="hero-logo">⚡ DropML</div>
    <div class="hero-tagline">Drop a CSV &nbsp;·&nbsp; Ask a Question &nbsp;·&nbsp; Get Answers</div>
    <p style="color:#6E7681;font-size:.87rem;margin:.2rem 0 .8rem;">
        AI analyzes your data, generates code, and explains the results — instantly.
    </p>
    <div class="hero-badge">🤖 Powered by Gemini &amp; Llama3 &nbsp;|&nbsp; Self-healing AI Engine</div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 1. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 .5rem;">
        <div style="font-size:1.4rem;font-weight:800;
                    background:linear-gradient(135deg,#88BDF2,#BDDDFC);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
            ⚡ DropML
        </div>
        <div style="font-size:.68rem;color:#484F58;letter-spacing:.1em;text-transform:uppercase;margin-top:2px;">
            Configuration
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="section-label">🧠 AI Engine</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Model", ("Gemini 1.5 Pro (Cloud)", "Llama 3 (Local API)"),
                                label_visibility="collapsed")

    api_key = ""
    if "Gemini" in model_choice:
        st.markdown('<div style="margin-top:.6rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🔑 API Key</div>', unsafe_allow_html=True)
        api_key = st.text_input("API Key", type="password", placeholder="AIza...",
                                label_visibility="collapsed")
        if api_key:
            configure_gemini(api_key)
            st.success("API Key configured ✓")

    st.divider()
    st.markdown("""
    <div style="background:rgba(136,189,242,.05);border:1px solid rgba(136,189,242,.1);
                border-radius:10px;padding:.85rem;margin-top:.3rem;">
        <div style="font-size:.76rem;color:#88BDF2;font-weight:600;margin-bottom:.4rem;">💡 Quick Tips</div>
        <div style="font-size:.73rem;color:#6E7681;line-height:1.65;">
            • For Llama 3, ensure Ollama is running at <code style="color:#88BDF2">localhost:11434</code><br>
            • Supports CSV, Excel (.xlsx, .xls), and JSON<br>
            • Try asking for charts, models, or stats
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 2. UPLOAD — single column, full width card
# ==========================================
st.markdown("""
<div class="card-wrap fade-in" style="animation-delay:.08s">
    <div class="section-label">📂 Step 1</div>
    <div class="step-header">Upload Your Data</div>
    <div class="step-sub">Drag &amp; drop or click to browse — CSV, Excel, or JSON</div>
</div>
""", unsafe_allow_html=True)

# File uploader lives OUTSIDE the HTML card (Streamlit limitation):
# we visually connect it with matching border-radius styling via CSS above.
uploaded_file = st.file_uploader(
    "Upload file",
    type=["csv", "xlsx", "xls", "json"],
    label_visibility="collapsed"
)

# ==========================================
# 3. POST-UPLOAD FLOW
# ==========================================
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        df = preprocess_dataframe(df)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # ── Data Preview ──────────────────────
    st.markdown("""
    <div class="card-wrap fade-in" style="animation-delay:.12s">
        <div class="section-label">📋 Preview</div>
        <div class="step-header">Data Table</div>
        <div class="step-sub">First 5 rows</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"`{uploaded_file.name}`  ·  {len(df):,} rows × {len(df.columns)} columns")
    st.dataframe(df.head(), use_container_width=True)

    # ── Catalog ───────────────────────────
    with st.spinner("Scanning data and generating catalog..."):
        catalog = generate_data_catalog(df)
        catalog_json_str = catalog_to_json(catalog)

    # ── Metric Dashboard ──────────────────
    num_cols   = sum(1 for c in catalog.get("columns", {}).values() if c.get("dtype","").startswith(("int","float")))
    cat_cols   = sum(1 for c in catalog.get("columns", {}).values() if not c.get("dtype","").startswith(("int","float")))
    null_count = sum(c.get("null_count", 0) for c in catalog.get("columns", {}).values())
    row_count  = catalog.get("row_count", len(df))
    col_count  = catalog.get("column_count", len(df.columns))
    null_color = "#f85149" if null_count > 0 else "#3fb950"

    st.markdown(f"""
    <div class="card-wrap fade-in" style="animation-delay:.18s">
        <div class="section-label">📊 Data Profile</div>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{row_count:,}</div>
                <div class="metric-label">Rows</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{col_count}</div>
                <div class="metric-label">Columns</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{num_cols}</div>
                <div class="metric-label">Numeric</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{cat_cols}</div>
                <div class="metric-label">Categorical</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color:{null_color}">{null_count}</div>
                <div class="metric-label">Null Values</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🗂️ View Full Data Catalog (Schema & Stats)"):
        st.json(catalog)

    # ==========================================
    # 4. PROMPT AREA — header card + widget below
    # ==========================================
    st.markdown("""
    <div class="card-wrap fade-in" style="animation-delay:.24s">
        <div class="section-label">🎯 Step 2</div>
        <div class="step-header">Ask Your Question</div>
        <div class="step-sub">Describe the analysis, visualization, or model you want to build</div>
    </div>
    """, unsafe_allow_html=True)

    # Text area is a normal Streamlit widget — CSS above gives it the glass style
    user_goal = st.text_area(
        "Analysis goal",
        placeholder=(
            "✨  What would you like to analyze?\n\n"
            "   e.g. Show the top 5 countries by employee count as a pie chart.\n"
            "   e.g. Build a classification model to predict churn.\n"
            "   e.g. Show the correlation heatmap between all numeric columns."
        ),
        height=150,
        label_visibility="collapsed"
    )

    # Tip + Button row
    col_hint, col_btn = st.columns([3, 1])
    with col_hint:
        st.markdown(
            '<p style="font-size:.76rem;color:#484F58;margin:0;padding-top:.45rem;">'
            '💡 Try: "feature importance", "correlation heatmap", "build a classifier"'
            '</p>',
            unsafe_allow_html=True
        )
    with col_btn:
        run_analysis = st.button("🚀 Analyze", type="primary", key="run_btn")

    # ==========================================
    # 5. EXECUTION ENGINE (logic unchanged)
    # ==========================================
    if run_analysis:
        if not user_goal.strip():
            st.warning("Please enter an analysis goal.")
            st.stop()
        if "Gemini" in model_choice and not api_key:
            st.warning("Please enter your Gemini API Key in the sidebar.")
            st.stop()

        from executor import run_generated_code, classify_error, build_self_healing_prompt

        st.markdown("""
        <div class="card-wrap fade-in" style="margin-top:1.5rem;">
            <div class="section-label">💡 Results</div>
            <div class="step-header">AI Execution Output</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Routing ──
        template_code = None
        template_id = None
        params = {}

        with st.spinner("🧠 Routing request to optimal execution path..."):
            goal_lower = user_goal.lower()
            _target_col = None
            for col in df.columns:
                if col.lower() in goal_lower:
                    _target_col = col
            if not _target_col:
                _target_col = df.columns[-1]

            if any(kw in goal_lower for kw in ["중요한 변수","변수 중요도","feature importance","important variable","importance"]):
                template_id = "feature_importance"; params = {"target_col": _target_col, "n": 5}
            elif any(kw in goal_lower for kw in ["예측","분류 모델","classification","predict","classifier"]):
                template_id = "classification_model"; params = {"target_col": _target_col}
            elif any(kw in goal_lower for kw in ["상관관계","correlation","히트맵","heatmap"]):
                template_id = "correlation_heatmap"; params = {}

            if not template_id:
                cls_prompt = build_classification_prompt(catalog_json_str, user_goal)
                cls_resp = generate_with_gemini(cls_prompt) if "Gemini" in model_choice else generate_with_ollama(cls_prompt, model_name="llama3")
                parsed = parse_classification_response(cls_resp)
                if parsed and parsed.get("template_id") and parsed["template_id"] != "none":
                    template_id = parsed["template_id"]; params = parsed.get("params", {})

            if template_id and template_id != "none":
                template_code = render_template(template_id, params, df_columns=df.columns.tolist())
                if template_code:
                    st.info(f"🎯 **Template Mode**: `{template_id}` pattern matched — using pre-validated code path.")

        # ── Template execution ──
        xai_success_code = None
        xai_output = ""

        if template_code:
            with st.spinner("Executing template code..."):
                success, final_code, error_trace = run_generated_code(template_code, df)
            if success:
                st.success("✅ Code executed successfully!")
                with st.expander("💻 View Executed Template Code"):
                    st.code(final_code, language="python")
                xai_success_code = final_code
                xai_output = error_trace if error_trace else "(Chart/visualization output complete)"
            else:
                st.warning("Template failed — switching to AI free-form generation...")
                template_code = None

        # ── Free-form + self-healing ──
        if not template_code:
            st.info("🔮 **Free-Form Mode** — AI is writing custom code for your request.")
            MAX_RETRIES = 3
            current_prompt = build_prompt(catalog_json_str, user_goal)

            for attempt in range(MAX_RETRIES):
                with st.spinner(f"[{model_choice}] Generating & executing code... (Attempt {attempt+1}/{MAX_RETRIES})"):
                    response_text = generate_with_gemini(current_prompt) if "Gemini" in model_choice else generate_with_ollama(current_prompt, model_name="llama3")
                    success, final_code, error_trace = run_generated_code(response_text, df)

                    if success:
                        st.success("✅ Code executed successfully!")
                        with st.expander("💻 View AI-Generated Code"):
                            st.code(final_code, language="python")
                        xai_success_code = final_code
                        xai_output = error_trace if error_trace else "(Chart/visualization output complete)"
                        break
                    else:
                        error_info = classify_error(error_trace)
                        st.warning(f"⚠️ Error detected — self-healing... (Attempt {attempt+1}/{MAX_RETRIES}) — `{error_info['type']}`")
                        with st.expander("View Error Details", expanded=False):
                            st.code(error_trace)
                            st.info(f"🔍 **Auto-Diagnosis:** {error_info['hint']}")
                        current_prompt = build_self_healing_prompt(final_code, error_trace, error_info)
            else:
                st.error(f"❌ Maximum retries ({MAX_RETRIES}) exceeded. Unable to auto-resolve the error.")

        # ── XAI ──
        if xai_success_code:
            with st.spinner("🧠 AI is interpreting the results..."):
                xai_prompt = build_xai_prompt(catalog_json_str, user_goal, xai_success_code, xai_output)
                xai_response = generate_with_gemini(xai_prompt) if "Gemini" in model_choice else generate_with_ollama(xai_prompt, model_name="llama3")

            st.markdown("""
            <div class="card-wrap fade-in" style="margin-top:1.5rem;">
                <div class="section-label">🧠 XAI</div>
                <div class="step-header">AI Analysis Interpretation</div>
                <div class="step-sub">Explainable AI breakdown of the results above</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(xai_response)
