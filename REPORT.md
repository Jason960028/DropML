# ?“‘ DropML Project Development Report

**Date**: April 4, 2026  
**Project**: Reproducing DropML ??Intelligent Data Catalog & ML Pipeline Generator  
**Tech Stack**: Python, Streamlit, Pandas, Scikit-learn, Matplotlib, Seaborn, Gemini API, Ollama (Llama 3)

---

## 1. Project Objective

Reproduce the core ideas from the DropML paper:
- Automatically generate a **Data Catalog** when a user uploads a CSV/Excel file
- Pass the catalog to an LLM to **auto-generate and execute analysis code** based on user requests
- Automatically recover from errors via a **Self-Healing** loop
- Provide evidence-based explanations of analysis decisions through **XAI** (Explainable AI)

---

## 2. System Architecture

```
?“‚ File Upload (CSV/XLSX)
  ???§¹ preprocessor.py ??7-Step Data Cleaning Pipeline
  ???“Š profiler.py ??Data Catalog Extraction (Schema + Statistics + Missing Values)
  ???§  app.py ??3-Tier Hybrid Routing
  ??  ?œâ? [Tier 1] Keyword Priority Matching (instant, no LLM call)
  ?œâ? [Tier 2] LLM Classification (template_id + params extraction)
  ?”â? [Tier 3] Free-Form Code Generation (LLM writes Python directly)
  ???“‹ templates.py ??10 Pre-Validated Code Templates + Fuzzy Matching
  ???›¡ï¸?executor.py ??3-Layer Defense + Error Classification + Self-Healing
  ???§  XAI ??Evidence-Based Analysis Explanation
```

---

## 3. File Structure

| File | Role | Lines |
|---|---|---|
| `app.py` | Streamlit UI + Hybrid Routing + XAI | ~210 |
| `executor.py` | Code Execution Engine (Parser ??Sandbox ??Self-Healing) | ~310 |
| `templates.py` | 10 Validated Analysis Templates + Fuzzy Column Matching | ~410 |
| `llm_client.py` | LLM Prompts (Generation / Classification / Self-Healing / XAI) | ~190 |
| `profiler.py` | Data Catalog Generator (Schema, Statistics, Missing Values) | ~90 |
| `preprocessor.py` | 7-Step Preprocessing Pipeline | ~50 |

---

## 4. Problems Encountered & Solutions

### ?”´ Problem 1: `matplotlib.style` ??FileNotFoundError

**Symptom**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'seaborn'
```

**Root Cause**: The LLM (Llama 3) generated `plt.style.use('seaborn')`. This style name was removed in recent versions of matplotlib.

**Solution**: Added a regex-based sanitizer in `executor.py` that automatically strips `plt.style.use()` calls from generated code before execution.

---

### ?”´ Problem 2: Scatter Plot ??RGBA ValueError

**Symptom**:
```
ValueError: RGBA values should be within 0-1 range
```

**Root Cause**: The LLM generated `plt.scatter(c=df['income'])`, passing string values (`<=50K`, `>50K`) directly as color arguments. Matplotlib cannot parse categorical strings as RGBA values.

**Solution**:
- Added a prompt rule: *"When coloring scatter plots by a categorical column, always use seaborn's `sns.scatterplot(hue=...)` instead of `plt.scatter`"*
- Implemented a pre-validated `scatter_plot` template using seaborn

---

### ?”´ Problem 3: `df.fillna(df.mean())` ??TypeError

**Symptom**:
```
TypeError: Cannot perform reduction 'mean' with string dtype
```

**Root Cause**: The LLM habitually inserted `df.fillna(df.mean())` as a "safety" measure. However, calling `mean()` on a DataFrame containing string columns raises a TypeError.

**Solution (Root Cause Elimination)**:
1. Added a **NaN elimination step** to `preprocessor.py` (numeric columns ??median, string columns ??'Unknown')
2. Updated LLM prompt: *"Data has NO NaN values. Do NOT use fillna() or dropna()"*
3. Added `TypeError`-specific error classification in `executor.py` with targeted Self-Healing hints

---

### ?”´ Problem 4: Column Name Mismatch ??KeyError

**Symptom**:
```
KeyError: "['hours-per-week', 'education-num'] not in index"
```

**Root Cause**: Users type `hours-per-week` (hyphens) but the actual column name loaded from Excel is `hours.per.week` (dots). The LLM extracted column names from user text instead of the Data Catalog.

**Solution**: Implemented a **Fuzzy Matching** system in `templates.py`:
```
Correction Process:
1. Exact match: "hours-per-week" == "hours.per.week"? ????2. Normalized match: "hoursperweek" == "hoursperweek"? ????Match!
3. Return: "hours.per.week" ??auto-corrected to actual column name
```

---

### ?”´ Problem 5: LLM Misclassification ??`feature_importance` classified as `bar_chart`

**Symptom**: The request "Show the top 5 most important variables" was classified as a simple bar chart instead of Feature Importance analysis.

**Root Cause**: The LLM was misled by the keyword "bar chart" in the request and returned `bar_chart`. The keyword safety net ran *after* LLM classification, so it couldn't override the wrong answer.

**Solution**: **Reversed the routing order** ??keyword matching now runs *before* LLM classification:
```
[Tier 1] Keyword Priority ??"important variables" detected ??feature_importance (instant)
[Tier 2] LLM Classification ??only handles requests not caught by keywords
[Tier 3] Free-Form Generation ??fallback when neither matches
```

---

### ?”´ Problem 6: `NameError: name 'parsed' is not defined`

**Symptom**: When keyword matching determined the template (skipping LLM classification), the `parsed` variable was never created. Downstream code referenced `parsed`, causing a crash.

**Solution**: Removed the unnecessary `parsed` reference and simplified the branching logic.

---

### ?”´ Problem 7: `string` dtype vs `object` dtype ??LabelEncoder Skipped

**Symptom**:
```
ValueError: could not convert string to float: 'Unknown'
```

**Root Cause**: The ML templates checked `model_df[c].dtype == 'object'` to detect categorical columns. However, pandas' newer `StringDtype` (`'string'`) is distinct from `'object'` ??the check missed these columns, leaving raw strings that sklearn cannot process.

**Solution**: Replaced the dtype check:
```python
# Before (bug): misses 'string' dtype
if model_df[c].dtype == 'object':

# After (fix): catches ALL non-numeric types
if not pd.api.types.is_numeric_dtype(model_df[c]):
```

---

## 5. Key Architecture Decisions

### 5.1 Hybrid Routing (Keyword ??LLM ??Free-Form)

| Decision | Rationale |
|---|---|
| Keyword matching runs before LLM | Prevents misclassification + faster response (no LLM call needed) |
| 10 pre-built templates | Compensates for small LLM (Llama 3) code generation limitations |
| Free-form mode retained | Handles creative requests outside template coverage |

### 5.2 3-Layer Defense System

| Layer | Purpose | Error Types Blocked |
|---|---|---|
| Layer 1: Parser | Code extraction + `compile()` syntax validation | SyntaxError |
| Layer 2: Sandbox | Pre-loaded library injection | ModuleNotFoundError |
| Layer 3: Self-Healing | Error classification + targeted fix hints | All runtime errors |

### 5.3 Error Classification System (8 Types)

| Type | Key Hint |
|---|---|
| `module_not_found` | Module already loaded in sandbox |
| `column_not_found` | Verify exact column names from catalog |
| `value_error_rgba` | Use seaborn's `hue` parameter instead |
| `zero_division` | Insert `np.where` guard |
| `key_error` | Check case-sensitivity and spaces |
| `str_accessor_error` | Verify column dtype before `.str` |
| `type_error_string` | No fillna needed; use `select_dtypes` |
| `type_error` | Avoid mixing string/numeric operations |

---

## 6. Validation Results

### Test Dataset: adult.xlsx (US Census Income Data, 32,561 rows Ã— 15 columns)

| # | Test Request | Routing | Result |
|---|---|---|---|
| 1 | Pie chart (occupation distribution) | ?Ž¯ Template | ??Pass |
| 2 | Bar chart (education vs hours-per-week) | ?Ž¯ Template | ??Pass |
| 3 | Histogram (age distribution) | ?Ž¯ Template | ??Pass |
| 4 | Value counts (marital status) | ?Ž¯ Template | ??Pass |
| 5 | Correlation heatmap | ?Ž¯ Template | ??Pass |
| 6 | Group statistics (race vs age) | ?Ž¯ Template | ??Pass |
| 7 | Top N values (highest hours-per-week) | ?Ž¯ Template | ??Pass |
| 8 | Classification model (income prediction) | ?Ž¯ Template | ??Pass (98.9%) |
| 9 | Feature importance (top 5 variables) | ?Ž¯ Keyword Match | ??Pass |
| 10 | Age-group income ratio (line chart) | ?”® Free-Form | ??Pass |

**Final Success Rate: 10/10 (100%)**

---

## 7. How to Run

```bash
# 1. Activate virtual environment
.\venv\Scripts\activate

# 2. Start Ollama (separate terminal)
ollama run llama3

# 3. Launch Streamlit app
streamlit run app.py
```
