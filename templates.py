"""
templates.py - 검증된 분석 코드 템플릿 라이브러리

사용자의 흔한 분석 요청을 사전에 검증된 안전한 코드로 처리합니다.
LLM은 "어떤 템플릿을 쓸지"와 "어떤 컬럼을 넣을지"만 판단하면 됩니다.
"""

# ============================================================
# 템플릿 레지스트리 (ID → 메타데이터 + 코드 생성 함수)
# ============================================================

TEMPLATE_REGISTRY = {
    "pie_chart": {
        "name": "파이 차트 (Top N)",
        "description": "특정 카테고리 컬럼의 상위 N개 항목을 파이 차트로 시각화",
        "required_params": ["category_col", "value_col", "n"],
    },
    "bar_chart": {
        "name": "막대 그래프 (그룹별 집계)",
        "description": "카테고리별로 수치 컬럼을 집계(합계/평균)하여 막대 그래프로 시각화",
        "required_params": ["category_col", "value_col", "agg_func", "n"],
    },
    "scatter_plot": {
        "name": "산점도 (두 수치 컬럼 관계)",
        "description": "두 수치 컬럼 간의 관계를 산점도로 시각화, 선택적으로 카테고리별 색상 구분",
        "required_params": ["x_col", "y_col", "hue_col"],
    },
    "histogram": {
        "name": "히스토그램 (분포)",
        "description": "수치 컬럼의 데이터 분포를 히스토그램으로 시각화",
        "required_params": ["target_col", "bins"],
    },
    "correlation_heatmap": {
        "name": "상관관계 히트맵",
        "description": "수치 컬럼들 간의 상관관계를 히트맵으로 시각화",
        "required_params": [],
    },
    "top_n_values": {
        "name": "상위/하위 N개 조회",
        "description": "특정 컬럼 기준으로 상위 또는 하위 N개 행을 조회",
        "required_params": ["sort_col", "n", "ascending"],
    },
    "group_stats": {
        "name": "그룹별 통계",
        "description": "카테고리별로 수치 컬럼의 통계(평균, 합계, 개수 등)를 계산",
        "required_params": ["group_col", "value_col", "agg_func"],
    },
    "classification_model": {
        "name": "분류 모델 (Random Forest)",
        "description": "카테고리 타겟을 예측하는 분류 모델을 학습하고 정확도를 출력",
        "required_params": ["feature_cols", "target_col"],
    },
    "feature_importance": {
        "name": "변수 중요도 시각화",
        "description": "분류/회귀 모델을 학습한 뒤 변수 중요도(Feature Importance)를 막대 그래프로 시각화",
        "required_params": ["feature_cols", "target_col", "n"],
    },
    "value_counts": {
        "name": "빈도수 계산",
        "description": "특정 컬럼의 고유값 빈도수를 계산하고 막대 그래프로 시각화",
        "required_params": ["target_col", "n"],
    },
}


def get_template_list_for_prompt() -> str:
    """LLM에게 전달할 템플릿 목록 텍스트를 생성합니다."""
    lines = []
    for tid, info in TEMPLATE_REGISTRY.items():
        params = ", ".join(info["required_params"]) if info["required_params"] else "(없음)"
        lines.append(f'- "{tid}": {info["name"]} — {info["description"]} [파라미터: {params}]')
    return "\n".join(lines)


# ============================================================
# 코드 생성 함수들 (각 템플릿별 검증된 파이썬 코드 반환)
# ============================================================

def generate_pie_chart(params: dict) -> str:
    cat = params.get("category_col", "category")
    val = params.get("value_col", None)
    n = params.get("n", 5)
    
    if val:
        return f"""
import matplotlib.pyplot as plt
grouped = df.groupby('{cat}')['{val}'].sum().reset_index()
grouped = grouped.sort_values(by='{val}', ascending=False).head({n})
fig, ax = plt.subplots(figsize=(6, 4))
ax.pie(grouped['{val}'], labels=grouped['{cat}'], autopct='%1.1f%%', startangle=90)
ax.set_title('Top {n} by {val} ({cat})')
plt.tight_layout()
import streamlit as st
st.pyplot(fig, clear_figure=True, use_container_width=False)
"""
    else:
        return f"""
import matplotlib.pyplot as plt
counts = df['{cat}'].value_counts().head({n})
fig, ax = plt.subplots(figsize=(6, 4))
ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
ax.set_title('Top {n} Distribution of {cat}')
plt.tight_layout()
import streamlit as st
st.pyplot(fig, clear_figure=True, use_container_width=False)
"""


def generate_bar_chart(params: dict) -> str:
    cat = params.get("category_col", "category")
    val = params.get("value_col", "value")
    agg = params.get("agg_func", "mean")
    n = params.get("n", 10)
    return f"""
import matplotlib.pyplot as plt
import seaborn as sns
grouped = df.groupby('{cat}')['{val}'].{agg}().reset_index()
grouped = grouped.sort_values(by='{val}', ascending=False).head({n})
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=grouped, x='{val}', y='{cat}', ax=ax, palette='viridis')
ax.set_title('Top {n} {cat} by {agg}({val})')
ax.set_xlabel('{val} ({agg})')
plt.tight_layout()
import streamlit as st
st.pyplot(fig, clear_figure=True, use_container_width=False)
"""


def generate_scatter_plot(params: dict) -> str:
    x = params.get("x_col", "x")
    y = params.get("y_col", "y")
    hue = params.get("hue_col", None)
    hue_part = f", hue='{hue}'" if hue and hue != "null" and hue != "None" else ""
    return f"""
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=df, x='{x}', y='{y}'{hue_part}, alpha=0.6, ax=ax)
ax.set_title('{y} vs {x}')
plt.tight_layout()
import streamlit as st
st.pyplot(fig, clear_figure=True, use_container_width=False)
"""


def generate_histogram(params: dict) -> str:
    col = params.get("target_col", "value")
    bins = params.get("bins", 30)
    return f"""
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(df['{col}'].dropna(), bins={bins}, kde=True, ax=ax, color='steelblue')
ax.set_title('Distribution of {col}')
ax.set_xlabel('{col}')
plt.tight_layout()
import streamlit as st
st.pyplot(fig, clear_figure=True, use_container_width=False)
"""


def generate_correlation_heatmap(params: dict) -> str:
    return """
import matplotlib.pyplot as plt
import seaborn as sns
numeric_df = df.select_dtypes(include='number')
corr = numeric_df.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, linewidths=0.5)
ax.set_title('Correlation Heatmap')
plt.tight_layout()
import streamlit as st
st.pyplot(fig, clear_figure=True, use_container_width=False)
"""


def generate_top_n_values(params: dict) -> str:
    col = params.get("sort_col", "value")
    n = params.get("n", 10)
    asc = params.get("ascending", False)
    return f"""
result = df.sort_values(by='{col}', ascending={asc}).head({n})
print(result.to_string())
import streamlit as st
st.dataframe(result)
"""


def generate_group_stats(params: dict) -> str:
    group = params.get("group_col", "category")
    val = params.get("value_col", "value")
    agg = params.get("agg_func", "mean")
    return f"""
result = df.groupby('{group}')['{val}'].{agg}().reset_index()
result = result.sort_values(by='{val}', ascending=False)
print(result.to_string())
import streamlit as st
st.dataframe(result)
"""


def generate_classification_model(params: dict) -> str:
    features = params.get("feature_cols", [])
    target = params.get("target_col", "target")
    feat_list = str(features)
    return f"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

feature_cols = {feat_list}
target_col = '{target}'

model_df = df[feature_cols + [target_col]].copy()

# 범주형 변수 인코딩 (모든 비숫자 컬럼에 적용)
le_dict = {{}}
for c in feature_cols:
    if not pd.api.types.is_numeric_dtype(model_df[c]):
        le = LabelEncoder()
        model_df[c] = le.fit_transform(model_df[c].astype(str))
        le_dict[c] = le

# 타겟 인코딩
le_target = LabelEncoder()
model_df[target_col] = le_target.fit_transform(model_df[target_col].astype(str))

X = model_df[feature_cols].values
y = model_df[target_col].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {{acc:.4f}} ({{acc*100:.1f}}%)")
print()
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

import streamlit as st
st.metric("Model Accuracy", f"{{acc*100:.1f}}%")
"""


def generate_feature_importance(params: dict) -> str:
    features = params.get("feature_cols", [])
    target = params.get("target_col", "target")
    n = params.get("n", 10)
    feat_list = str(features)
    return f"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

feature_cols = {feat_list}
target_col = '{target}'

model_df = df[feature_cols + [target_col]].copy()

# 범주형 변수 인코딩 (모든 비숫자 컬럼에 적용)
for c in feature_cols:
    if not pd.api.types.is_numeric_dtype(model_df[c]):
        le = LabelEncoder()
        model_df[c] = le.fit_transform(model_df[c].astype(str))

le_target = LabelEncoder()
model_df[target_col] = le_target.fit_transform(model_df[target_col].astype(str))

X = model_df[feature_cols].values
y = model_df[target_col].values

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:{n}]

fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(range(len(indices)), importances[indices], align='center', color='steelblue')
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([feature_cols[i] for i in indices])
ax.set_xlabel('Importance')
ax.set_title('Top {n} Feature Importance')
ax.invert_yaxis()
plt.tight_layout()
import streamlit as st
st.pyplot(fig, clear_figure=True, use_container_width=False)
"""


def generate_value_counts(params: dict) -> str:
    col = params.get("target_col", "category")
    n = params.get("n", 10)
    return f"""
import matplotlib.pyplot as plt
import seaborn as sns
counts = df['{col}'].value_counts().head({n}).reset_index()
counts.columns = ['{col}', 'count']
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=counts, x='count', y='{col}', ax=ax, palette='viridis')
ax.set_title('Top {n} Value Counts: {col}')
plt.tight_layout()
import streamlit as st
st.pyplot(fig, clear_figure=True, use_container_width=False)
"""


# ============================================================
# 컬럼명 자동 교정 (Fuzzy Matching)
# ============================================================

def _normalize(name: str) -> str:
    """컬럼명을 정규화하여 비교할 수 있도록 합니다. (하이픈, 점, 언더스코어 → 통일)"""
    return name.lower().replace("-", "").replace(".", "").replace("_", "").replace(" ", "")

def _fuzzy_match_column(name: str, actual_columns: list) -> str:
    """
    LLM이 추출한 컬럼명이 실제 DataFrame에 없을 때,
    가장 유사한 컬럼명을 자동으로 찾아 반환합니다.
    
    예: 'hours-per-week' → 'hours.per.week' (정규화 매칭)
    """
    # 1. 정확히 일치하면 그대로 반환
    if name in actual_columns:
        return name
    
    # 2. 정규화 후 비교 (하이픈/점/언더스코어/공백 모두 무시)
    normalized_input = _normalize(name)
    for col in actual_columns:
        if _normalize(col) == normalized_input:
            return col
    
    # 3. 부분 문자열 매칭 (긴 이름의 일부가 맞는 경우)
    for col in actual_columns:
        if normalized_input in _normalize(col) or _normalize(col) in normalized_input:
            return col
    
    # 4. 매칭 실패 → 원본 반환 (에러는 실행 시 처리)
    return name


def fix_column_names(params: dict, df_columns: list) -> dict:
    """
    params 딕셔너리 안의 모든 컬럼명을 실제 DataFrame의 컬럼명과 대조하여 자동 교정합니다.
    """
    column_param_keys = [
        "category_col", "value_col", "x_col", "y_col", "hue_col",
        "target_col", "sort_col", "group_col"
    ]
    actual = list(df_columns)
    fixed = dict(params)
    
    for key in column_param_keys:
        if key in fixed and fixed[key] and fixed[key] not in ("null", "None", None):
            fixed[key] = _fuzzy_match_column(str(fixed[key]), actual)
    
    # feature_cols는 리스트이므로 별도 처리
    if "feature_cols" in fixed and isinstance(fixed["feature_cols"], list):
        fixed["feature_cols"] = [_fuzzy_match_column(str(c), actual) for c in fixed["feature_cols"]]
    
    return fixed


# ============================================================
# 템플릿 디스패처 (template_id → 코드 생성 함수 매핑)
# ============================================================

_GENERATORS = {
    "pie_chart": generate_pie_chart,
    "bar_chart": generate_bar_chart,
    "scatter_plot": generate_scatter_plot,
    "histogram": generate_histogram,
    "correlation_heatmap": generate_correlation_heatmap,
    "top_n_values": generate_top_n_values,
    "group_stats": generate_group_stats,
    "classification_model": generate_classification_model,
    "feature_importance": generate_feature_importance,
    "value_counts": generate_value_counts,
}


def render_template(template_id: str, params: dict, df_columns: list = None) -> str:
    """
    주어진 template_id와 params로 검증된 파이썬 코드를 생성합니다.
    df_columns가 제공되면 컬럼명 자동 교정을 수행합니다.
    
    Returns:
        생성된 파이썬 코드 문자열. template_id가 유효하지 않으면 None.
    """
    generator = _GENERATORS.get(template_id)
    if generator is None:
        return None
    
    # 컬럼명 자동 교정 적용
    if df_columns is not None:
        params = fix_column_names(params, df_columns)
    
    # feature_importance에서 feature_cols가 비어있으면 모든 수치 컬럼을 자동 선택
    if template_id in ("feature_importance", "classification_model"):
        if not params.get("feature_cols") or params["feature_cols"] == []:
            # df_columns에서 target_col을 제외한 나머지를 feature로 사용
            # (실제 수치 컬럼 필터링은 코드 내부에서 수행)
            target = params.get("target_col", "")
            params["feature_cols"] = [c for c in (df_columns or []) if c != target]
    
    return generator(params)

