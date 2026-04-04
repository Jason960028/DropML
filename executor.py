"""
executor.py - 3단계 방어벽 아키텍처 (Production-Grade Code Executor)

Layer 1: 파서 강화 (코드 추출 & 구문 검증)
Layer 2: 샌드박스 강화 (모든 라이브러리 사전 주입)
Layer 3: 스마트 Self-Healing (에러 유형별 맞춤 수정 지시)
"""
import re
import ast
import traceback
import streamlit as st

# ============================================================
# Layer 2: 샌드박스 강화 - 모든 데이터 과학 라이브러리 사전 로딩
# ============================================================
# AI가 어떤 라이브러리를 import하든 에러가 나지 않도록,
# 실행 환경에 미리 모든 라이브러리를 적재해둡니다.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI 없는 서버 환경에서도 안전하게 렌더링
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn 주요 모듈 사전 로딩
from sklearn import preprocessing, model_selection, metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

def _build_sandbox_namespace(df):
    """
    AI가 생성한 코드가 실행될 샌드박스 네임스페이스를 구성합니다.
    여기에 등록된 모든 라이브러리와 모듈은 import 없이 바로 사용 가능합니다.
    """
    namespace = {
        # 기본 내장 함수 (print, len, range, enumerate, zip 등)
        "__builtins__": __builtins__,
        
        # 데이터 & 수학
        "df": df,
        "pd": pd,
        "np": np,
        
        # 시각화
        "plt": plt,
        "sns": sns,
        "matplotlib": matplotlib,
        
        # Streamlit (plt.show 변환 후 사용됨)
        "st": st,
        
        # Scikit-Learn 전체
        "sklearn": __import__('sklearn'),
        "preprocessing": preprocessing,
        "model_selection": model_selection,
        "metrics": metrics,
        "LinearRegression": LinearRegression,
        "LogisticRegression": LogisticRegression,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "RandomForestClassifier": RandomForestClassifier,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "KMeans": KMeans,
        "PCA": PCA,
        "KNeighborsClassifier": KNeighborsClassifier,
        "train_test_split": model_selection.train_test_split,
        "cross_val_score": model_selection.cross_val_score,
        "accuracy_score": metrics.accuracy_score,
        "classification_report": metrics.classification_report,
        "confusion_matrix": metrics.confusion_matrix,
        "mean_squared_error": metrics.mean_squared_error,
        "r2_score": metrics.r2_score,
        "LabelEncoder": preprocessing.LabelEncoder,
        "StandardScaler": preprocessing.StandardScaler,
        "MinMaxScaler": preprocessing.MinMaxScaler,
    }
    return namespace


# ============================================================
# Layer 1: 파서 강화 - 코드 추출 & 구문 사전 검증
# ============================================================

def parse_code(llm_response: str) -> str:
    """LLM의 응답에서 파이썬 코드 부분만 추출합니다."""
    # 정규식 패턴을 사용하여 ```python ... ``` 또는 ``` ... ``` 블록 추출
    pattern = r"```(?:python)?\s*\n?(.*?)```"
    matches = re.findall(pattern, llm_response, re.DOTALL | re.IGNORECASE)
    
    if matches:
        code = matches[0].strip()
    else:
        code = llm_response.strip()
        
    # 간혹 코드 첫 줄에 'python' 텍스트만 남아있는 경우가 있어 예외 처리
    lines = code.split('\n')
    if lines and lines[0].strip().lower() == 'python':
        code = '\n'.join(lines[1:]).strip()
        
    return code


def sanitize_code(code: str) -> str:
    """
    코드를 실행 환경(Streamlit)에 맞게 변환하고,
    흔한 문제 패턴들을 사전에 교정합니다.
    """
    # 1. plt.show()를 Streamlit 호환 코드로 변환
    show_replacement = (
        "fig = plt.gcf()\n"
        "st.pyplot(fig, clear_figure=True, use_container_width=False)"
    )
    code = code.replace("plt.show()", show_replacement)
    
    # 2. 더 이상 필요 없는 중복 import 제거 (샌드박스에 이미 모두 주입됨)
    #    단, 제거하면 안 되는 from sklearn.xxx import yyy 같은 서브모듈 임포트는 유지
    safe_imports = [
        r"^import pandas as pd\s*$",
        r"^import numpy as np\s*$",
        r"^import matplotlib\.pyplot as plt\s*$",
        r"^import matplotlib\s*$",
        r"^import seaborn as sns\s*$",
        r"^import streamlit as st\s*$",
        r"^from matplotlib import .*$",
    ]
    for pattern in safe_imports:
        code = re.sub(pattern, "# (auto-removed: already in sandbox)", code, flags=re.MULTILINE)

    # 3. pd.np (deprecated) → np 로 자동 교정
    code = code.replace("pd.np.", "np.")
    
    # 4. plt.style.use('seaborn') → sns.set_theme() 로 교정 (matplotlib 3.x 호환)
    code = re.sub(
        r"plt\.style\.use\(['\"]seaborn['\"].*?\)", 
        "sns.set_theme()", 
        code
    )
    
    return code


def validate_syntax(code: str) -> tuple:
    """
    exec() 실행 전에 Python의 compile()로 문법(Syntax)을 사전 검증합니다.
    문법 에러가 있으면 (False, 에러메시지)를, 없으면 (True, None)을 반환합니다.
    """
    try:
        compile(code, "<generated>", "exec")
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}\n  → {e.text}"


# ============================================================
# Layer 3: 스마트 Self-Healing - 에러 유형별 맞춤 수정 지시
# ============================================================

def classify_error(error_trace: str) -> dict:
    """
    에러 트레이스백을 분석하여 에러 유형을 분류하고,
    AI에게 전달할 맞춤형 수정 힌트를 생성합니다.
    """
    result = {"type": "unknown", "hint": ""}
    
    if "ModuleNotFoundError" in error_trace:
        # 어떤 모듈이 없는지 추출
        match = re.search(r"No module named '(\w+)'", error_trace)
        module_name = match.group(1) if match else "unknown"
        result["type"] = "missing_module"
        result["hint"] = (
            f"The module '{module_name}' is not available in this environment. "
            f"Available libraries are: pandas(pd), numpy(np), matplotlib.pyplot(plt), seaborn(sns), "
            f"and sklearn (with submodules: LinearRegression, LogisticRegression, DecisionTreeClassifier, "
            f"RandomForestClassifier, KMeans, PCA, LabelEncoder, StandardScaler, train_test_split, "
            f"accuracy_score, etc.). Rewrite using ONLY these available libraries."
        )
    elif "ValueError" in error_trace and "RGBA" in error_trace:
        result["type"] = "color_value_error"
        result["hint"] = (
            "You passed a categorical string column as the 'c' (color) argument to plt.scatter(). "
            "This does NOT work. Instead, use seaborn: `sns.scatterplot(data=df, x='col1', y='col2', hue='category_col')`. "
            "Do NOT use plt.scatter when coloring by categorical variables."
        )
    elif "ZeroDivisionError" in error_trace:
        result["type"] = "zero_division"
        result["hint"] = (
            "A ZeroDivisionError occurred. Before any division or percentage calculation, "
            "add a check: `if denominator != 0:` or use `np.where(denominator != 0, numerator/denominator, 0)`."
        )
    elif "KeyError" in error_trace:
        match = re.search(r"KeyError: ['\"](.+?)['\"]", error_trace)
        key_name = match.group(1) if match else "unknown"
        result["type"] = "key_error"
        result["hint"] = (
            f"Column '{key_name}' was not found in the DataFrame. "
            f"Check the Data Catalog carefully for exact column names (case-sensitive, no extra spaces). "
            f"Use `df.columns.tolist()` mentally to verify."
        )
    elif "AttributeError" in error_trace and ".str" in error_trace:
        result["type"] = "str_accessor_error"
        result["hint"] = (
            "You used .str accessor on a non-string column. "
            "Check the column dtype first. Only use .str on columns with dtype 'object' or 'string'. "
            "For numeric columns, use direct comparison or pd.to_numeric()."
        )
    elif "SyntaxError" in error_trace:
        result["type"] = "syntax_error"
        result["hint"] = (
            "There is a syntax error in the generated code. Common causes: "
            "unclosed quotes, missing colons, unmatched parentheses. "
            "Please carefully review and fix the Python syntax."
        )
    elif "TypeError" in error_trace and "string dtype" in error_trace:
        result["type"] = "type_error_string"
        result["hint"] = (
            "You tried to perform a numeric operation (like mean, sum) on string columns. "
            "IMPORTANT: The data is already pre-cleaned with NO NaN values remaining. "
            "Do NOT use df.fillna(df.mean()) or df.mean() on the full dataframe. "
            "If you need numeric operations, select only numeric columns first: "
            "df.select_dtypes(include='number'). "
            "The data has NO missing values - skip all fillna/dropna steps entirely."
        )
    elif "TypeError" in error_trace:
        result["type"] = "type_error"
        result["hint"] = (
            "A TypeError occurred, likely due to mixing string and numeric operations. "
            "The data is pre-cleaned with NO NaN values. Do NOT call fillna() or dropna(). "
            "Use df.select_dtypes(include='number') for numeric operations."
        )
    else:
        result["type"] = "runtime_error"
        result["hint"] = (
            "A runtime error occurred. Review the traceback carefully and fix the root cause. "
            "IMPORTANT: The data is already fully pre-cleaned: no leading/trailing spaces, "
            "no NaN values (numeric filled with median, strings filled with 'Unknown'), "
            "and non-standard nulls like '?' are already handled. "
            "Do NOT use fillna() or dropna() — they are unnecessary."
        )
    
    return result


def build_self_healing_prompt(original_code: str, error_trace: str, error_info: dict) -> str:
    """에러 유형에 맞는 정밀한 Self-Healing 프롬프트를 구성합니다."""
    return f"""You previously generated Python code that caused an error during execution.

[Your Code]
```python
{original_code}
```

[Error Traceback]
```
{error_trace}
```

[Error Analysis & Fix Hint]
Error Type: {error_info['type']}
Fix Guidance: {error_info['hint']}

[Available Pre-loaded Libraries in Sandbox]
pandas(pd), numpy(np), matplotlib.pyplot(plt), seaborn(sns), sklearn modules 
(LinearRegression, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, 
KMeans, PCA, LabelEncoder, StandardScaler, train_test_split, accuracy_score, etc.)
These are already imported - you do NOT need to import them again.

Please fix the error and provide the COMPLETE updated Python code.
Return ONLY valid Python code inside ```python ... ``` block.
Assume data is in DataFrame `df`. Do not reload the data.
"""


# ============================================================
# 메인 실행 함수
# ============================================================

def run_generated_code(code: str, df):
    """
    3단계 방어벽을 거쳐 AI 생성 코드를 안전하게 실행합니다.
    
    Returns:
        (success: bool, final_code: str, error_trace: str or None)
    """
    # --- Layer 1: 코드 추출 및 정제 ---
    parsed_code = parse_code(code)
    sanitized_code = sanitize_code(parsed_code)
    
    # --- Layer 1.5: 구문 사전 검증 ---
    is_valid, syntax_error = validate_syntax(sanitized_code)
    if not is_valid:
        return False, sanitized_code, f"[Syntax Validation Failed]\n{syntax_error}"
    
    # --- Layer 2: 샌드박스 환경에서 실행 ---
    sandbox = _build_sandbox_namespace(df)
    
    try:
        exec(sanitized_code, sandbox)
        return True, sanitized_code, None
    except Exception as e:
        error_trace = traceback.format_exc()
        return False, sanitized_code, error_trace
