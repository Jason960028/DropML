import os
import json
import requests
import google.generativeai as genai
from templates import get_template_list_for_prompt

def configure_gemini(api_key: str):
    """Register your Gemini API key. (Typically obtained from the Streamlit interface or the environment variables.)"""
    genai.configure(api_key=api_key)

def generate_with_gemini(prompt: str, model_name: str = "gemini-1.5-pro") -> str:
    """
    Generate code using the Google Gemini API.
    (Performance Mode - Ideal for powerful inference and leveraging complex, up-to-date libraries)
    """
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Gemini API Error] {str(e)}"

def generate_with_ollama(prompt: str, model_name: str = "llama3", host: str = "http://localhost:11434") -> str:
    """
    Generate code using the local Ollama.
    (Privacy Mode - Ideal when you don't want to send data to the cloud)
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
        return "[Ollama Error] Failed to connect to the local server (localhost:11434). Please make sure the Ollama app is running."
    except Exception as e:
        return f"[Ollama Error] {str(e)}"

def build_prompt(data_catalog_json: str, user_goal: str) -> str:
    """
    [Free Code Generation Mode] Combine the data catalog with your objectives to construct a prompt for the LLM.
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
    [Hybrid Mode] Instruct the LLM to classify the user's request and extract only the parameters.
    Since it doesn't involve writing code—only returning JSON—even a small model (Llama 3) can handle this task effectively.
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
    Extracts and parses JSON from the LLM's classification response.
    Returns `None` on failure to switch to free-form code generation mode (Fallback).
    """
    import re
    
    
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        
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
    Generates XAI (Explainable AI) prompts.
    After the analysis is successful, it asks the LLM to explain why it chose this analysis method.
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

Please provide a clear, concise explanation in **English** following this exact structure:

## 📋 Analysis Summary
(Summarize the analysis performed in one sentence)

## 🧠 Why was this method chosen?
(Explain in 2–3 lines why this analysis method is appropriate, based on statistics from the data catalog.
Example: “Because the standard deviation of the ‘age’ column is X, indicating high variance...”, “Since the ‘income’ column is categorical...”)

## 📊 Interpretation of Results
(Interpret the key findings in 1–3 sentences so that the user can understand the results)

## 💡 Suggested Next Steps
(Suggest 1–2 additional analyses that could be conducted based on these results)

Keep it under 200 words total. Use bullet points where appropriate.
"""
