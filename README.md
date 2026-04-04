<div align="center">

# ⚡ DropML

### Drop a CSV. Ask a Question. Get Answers.

**An AI-powered data analysis tool that reads your data, understands its structure,<br>and generates & executes ML pipelines — all from a single natural language prompt.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

**Upload** → **Profile** → **Analyze** → **Explain**

</div>

## ✨ What is DropML?

DropML turns raw datasets into actionable insights **without writing a single line of code**.

1. 📂 **Upload** any CSV or Excel file
2. 🔍 **DropML auto-generates a Data Catalog** — schema, statistics, distributions, missing values
3. 💬 **Ask anything in plain language** — *"Predict income based on age and education"*
4. ⚡ **AI generates, executes, and self-heals** Python code in real-time
5. 🧠 **XAI explains** *why* that analysis method was chosen, backed by your data's statistics

> *"Show me the top 5 most important variables for predicting income"*
>
> → DropML auto-selects Random Forest, encodes categorical features, trains the model,
> plots feature importance, and explains the reasoning — **in under 10 seconds.**

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/DropML.git
cd DropML

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Start the local LLM (separate terminal)
ollama run llama3

# Launch DropML
streamlit run app.py
```

Open `http://localhost:8501` and drop your first CSV. That's it.

---

## 🏗️ Architecture

```
📂 File Upload
  ↓
🧹 7-Step Preprocessing ─── NaN elimination, type coercion, whitespace cleanup
  ↓
📊 Auto Data Catalog ────── Schema + Statistics + Missing Value Analysis
  ↓
🧠 3-Tier Hybrid Router
  ├─ [1] Keyword Matching ── Instant routing, no LLM call
  ├─ [2] LLM Classification ── AI picks the right template
  └─ [3] Free-Form Generation ── AI writes custom Python code
  ↓
🛡️ 3-Layer Defense
  ├─ Parser ───── Syntax validation via compile()
  ├─ Sandbox ──── Pre-loaded numpy, pandas, sklearn, seaborn
  └─ Self-Heal ── Error classification → targeted fix → retry
  ↓
🧠 XAI ── "Here's WHY I chose this method, based on your data."
```

---

## 📋 Supported Analysis Types

| Category | Analysis | Mode |
|---|---|---|
| 📊 **Visualization** | Pie Chart, Bar Chart, Histogram, Scatter Plot, Heatmap | Template |
| 📈 **Statistics** | Group Aggregation, Value Counts, Top-N Query | Template |
| 🤖 **Machine Learning** | Random Forest Classification, Feature Importance | Template |
| 🔮 **Custom** | Any analysis describable in natural language | Free-Form AI |

---

## 🛡️ Self-Healing Engine

DropML doesn't just generate code — it **guarantees execution**.

When AI-generated code fails, the Self-Healing engine:

1. **Classifies** the error into 8 distinct types (KeyError, TypeError, ValueError, etc.)
2. **Generates targeted hints** specific to the error type
3. **Retries with context** — the LLM receives the error trace + fix guidance
4. **Auto-corrects column names** via fuzzy matching (`hours-per-week` → `hours.per.week`)

```
Error: KeyError "hours-per-week"
  → Fuzzy Match: normalized "hoursperweek" = "hoursperweek" ✓
  → Auto-corrected to "hours.per.week"
  → Re-executed successfully
```

---

## 🧠 Explainable AI (XAI)

Every successful analysis comes with an AI-generated explanation:

```
📋 Summary
  Built a Random Forest classifier to predict income using age,
  education-num, and hours-per-week.

🧠 Why This Method?
  The income column has 2 categories (<=50K, >50K), making this
  a binary classification problem. Age has a standard deviation of
  13.64, indicating high variance — a strong candidate for
  tree-based models.

📊 Result Interpretation
  Model accuracy: 98.9%. Education level and age are the
  strongest predictors of income.

💡 Next Steps
  Try adding capital-gain and occupation as features to
  potentially improve accuracy.
```

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **LLM (Cloud)** | Google Gemini API |
| **LLM (Local)** | Ollama + Llama 3 |

---

## 📁 Project Structure

```
DropML/
├── app.py              # Main application — UI + hybrid routing + XAI
├── executor.py         # Code execution engine — sandbox + self-healing
├── templates.py        # 10 pre-validated analysis templates + fuzzy matching
├── llm_client.py       # LLM prompt management (Gemini / Ollama)
├── profiler.py         # Auto data catalog generator
├── preprocessor.py     # 7-step data cleaning pipeline
├── requirements.txt    # Python dependencies
├── REPORT.md           # Development report
└── README.md           # This file
```

---

## 🔑 Configuration

### Option A: Local LLM (Free, Private)
```bash
# Install Ollama: https://ollama.com
ollama pull llama3
ollama run llama3
```
Select **"🦙 Ollama (Llama 3)"** in the sidebar.

### Option B: Gemini API (Faster, More Accurate)
1. Get your API key from [Google AI Studio](https://aistudio.google.com)
2. Enter the key in the sidebar input field
3. Select **"✨ Gemini"**

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with curiosity and a lot of debugging.**

⭐ Star this repo if DropML saved you from writing boilerplate analysis code.

</div>
