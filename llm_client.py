import os
import json
import requests
import google.generativeai as genai
from templates import get_template_list_for_prompt

def configure_gemini(api_key: str):
    """Gemini API 키를 등록합니다. (보통 Streamlit 화면 또는 env에서 받음)"""
    genai.configure(api_key=api_key)

def generate_with_gemini(prompt: str, model_name: str = "gemini-1.5-pro") -> str:
    """
    구글 Gemini API를 활용하여 코드를 생성합니다.
    (Performance Mode - 강력한 추론 및 복잡한 최신 라이브러리 활용 시 적합)
    """
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Gemini API Error] {str(e)}"

def generate_with_ollama(prompt: str, model_name: str = "llama3", host: str = "http://localhost:11434") -> str:
    """
    로컬 Ollama를 활용하여 코드를 생성합니다.
    (Privacy Mode - 클라우드로 데이터 전송을 원치 않을 때 적합)
    """
    url = f"{host}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except requests.exceptions.ConnectionError:
        return "[Ollama Error] 로컬 서버(localhost:11434)에 연결 실패. Ollama 앱이 켜져 있는지 확인하세요."
    except Exception as e:
        return f"[Ollama Error] {str(e)}"

def build_prompt(data_catalog_json: str, user_goal: str) -> str:
    """
    [자유 코드 생성 모드] 데이터 카탈로그와 사용자 목표를 합쳐 LLM에게 던질 프롬프트를 구성합니다.
    """
    prompt = f"""You are an elite Data Scientist and Machine Learning Engineer.
Below is the Data Catalog (Schema, missing values, top categories, and stats) of a target dataset.

[Data Catalog]
{data_catalog_json}

[User Goal]
{user_goal}

Using Python, Pandas, and Scikit-Learn (or any applicable library), write a complete, executable Python code script that achieves the User Goal.
Assume the data is already loaded into a pandas DataFrame named `df`. DO NOT write code to load the csv file.

Strict Requirements:
1. Return ONLY the valid Python code. Do not provide any conversational text before or after the code.
2. Put the code strictly within ```python ... ``` block.
3. CRITICAL: The data is ALREADY fully cleaned. There are NO missing/NaN values. Do NOT use fillna(), dropna(), or df.mean() on the full dataframe. Skip all data cleaning steps.
4. Print the final results (e.g., metric, shape, or top values) nicely to `sys.stdout`.
5. If creating plots using matplotlib, ensure you use a moderate figsize (e.g., `plt.figure(figsize=(6, 4))`) so it fits seamlessly on the screen.
6. Important: Always use `import numpy as np` if you need numpy. Never use `pd.np` as it is deprecated.
7. Important: If coloring a scatter plot by a categorical string column, prefer using `seaborn` (e.g., `sns.scatterplot(..., hue='category')`) over `plt.scatter` to avoid RGBA ValueError.
8. Important: Always guard against `ZeroDivisionError` when calculating averages or percentages.
9. Important: For encoding categorical columns for ML, use LabelEncoder on each column individually: `le = LabelEncoder(); df[col] = le.fit_transform(df[col].astype(str))`.
"""
    return prompt


def build_classification_prompt(data_catalog_json: str, user_goal: str) -> str:
    """
    [하이브리드 모드] LLM에게 사용자의 요청을 분류하고 파라미터만 추출하도록 지시합니다.
    코드를 짜는 것이 아니라, JSON만 반환하면 되므로 작은 모델(Llama 3)도 잘 수행합니다.
    """
    template_list = get_template_list_for_prompt()
    
    prompt = f"""You are a request classifier. Your job is to analyze the user's goal and decide which pre-built template to use.

[Data Catalog]
{data_catalog_json}

[User Goal]
{user_goal}

[Available Templates]
{template_list}

[Instructions]
1. Choose the BEST matching template_id from the list above.
2. Extract the required parameters by looking at the Data Catalog's column names.
3. If NO template matches the user's goal, set template_id to "none".
4. Return ONLY a JSON object. No other text.

[Parameter Guidelines]
- "n": number of items (default 5)
- "agg_func": one of "sum", "mean", "count", "max", "min"
- "ascending": true for ascending, false for descending
- "feature_cols": list of column names (strings) to use as features
- "hue_col": set to "null" if no color grouping is needed
- "bins": number of histogram bins (default 30)

[Response Format - return ONLY this JSON, nothing else]
```json
{{
  "template_id": "<template_id or none>",
  "params": {{
    "<param_name>": "<value>",
    ...
  }}
}}
```
"""
    return prompt


def parse_classification_response(response_text: str) -> dict:
    """
    LLM의 분류 응답에서 JSON을 추출하고 파싱합니다.
    실패 시 None을 반환하여 자유 코드 생성 모드로 전환합니다 (Fallback).
    """
    import re
    
    # JSON 블록 추출 시도 (```json ... ``` 또는 { ... })
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # 중괄호로 시작하는 JSON 직접 찾기
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0).strip()
        else:
            return None
    
    try:
        result = json.loads(json_str)
        if "template_id" not in result:
            return None
        return result
    except (json.JSONDecodeError, Exception):
        return None


def build_xai_prompt(catalog_json: str, user_goal: str, executed_code: str, execution_output: str) -> str:
    """
    XAI(설명 가능한 AI) 프롬프트를 생성합니다.
    분석이 성공한 후, LLM에게 '왜 이 분석 방식을 선택했는지' 설명을 요청합니다.
    """
    return f"""You are a data science advisor. A user uploaded a dataset and requested an analysis.
The analysis has been completed successfully. Now you must EXPLAIN the analysis to the user.

[Data Catalog]
{catalog_json}

[User's Goal]
{user_goal}

[Executed Code]
```python
{executed_code}
```

[Execution Output]
{execution_output}

Please provide a clear, concise explanation in **Korean** following this exact structure:

## 📋 분석 요약
(한 문장으로 어떤 분석을 수행했는지 요약)

## 🧠 왜 이 방법을 선택했는가?
(데이터 카탈로그의 통계치를 근거로 이 분석 방법이 적합한 이유를 2-3줄로 설명.
예: "age 컬럼의 표준편차가 X로 분산이 크기 때문에...", "income 컬럼이 범주형이므로...")

## 📊 결과 해석
(실행 결과를 사용자가 이해할 수 있도록 핵심만 1-3줄로 해석)

## 💡 다음 단계 제안
(이 분석 결과를 바탕으로 추가로 해볼 수 있는 분석 1-2가지를 제안)

Keep it under 200 words total. Use bullet points where appropriate.
"""
