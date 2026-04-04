import pandas as pd
import numpy as np
import json

def generate_data_catalog(df: pd.DataFrame, max_unique_vals: int = 5, max_sample_rows: int = 100000) -> dict:
    """
    Pandas DataFrame을 입력받아 Schema, 통계량, 결측치 정보가 포함된 Data Catalog를 생성합니다.
    이 결과물은 이후 LLM(Gemini / Llama 3)에게 전달할 데이터의 '설명서' 역할을 합니다.
    """
    original_rows = len(df)
    
    # 🌟 대용량 데이터 샘플링 안전장치
    if original_rows > max_sample_rows:
        df = df.sample(n=max_sample_rows, random_state=42)
        
    catalog = {
        "num_rows": original_rows,
        "sampled_rows": len(df),
        "num_columns": len(df.columns),
        "columns": {}
    }
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        # 통계량 연산 속도를 위해 샘플링된 데이터(df)를 기준으로 결측치를 산출합니다.
        missing_count = int(df[col].isnull().sum())
        missing_ratio = float(missing_count / len(df)) * 100
        
        col_info = {
            "dtype": dtype,
            "missing_count_in_sample": missing_count,
            "missing_ratio_percent": round(missing_ratio, 2)
        }
        
        # 1) 날짜형 데이터 (Datetime) - 수치형/문자열보다 우선 판별 필요
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            col_info.update({
                "min_date": str(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max_date": str(df[col].max()) if not pd.isna(df[col].max()) else None,
            })
            
        # 2) 수치형 데이터 (Numeric)
        elif pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            col_info.update({
                "mean": float(desc.get("mean", 0)) if not pd.isna(desc.get("mean")) else None,
                "std": float(desc.get("std", 0)) if not pd.isna(desc.get("std")) else None,
                "min": float(desc.get("min", 0)) if not pd.isna(desc.get("min")) else None,
                "25%": float(desc.get("25%", 0)) if not pd.isna(desc.get("25%")) else None,
                "50%": float(desc.get("50%", 0)) if not pd.isna(desc.get("50%")) else None,
                "75%": float(desc.get("75%", 0)) if not pd.isna(desc.get("75%")) else None,
                "max": float(desc.get("max", 0)) if not pd.isna(desc.get("max")) else None,
            })
            
        # 3) 범주형/문자열 데이터 (Object/Category)
        else:
            num_unique = df[col].nunique()
            col_info.update({
                "num_unique": int(num_unique),
                # 고유값이 너무 많으면 LLM 컨텍스트 한도를 넘을 수 있으므로 상위 N개만 추출
                "top_values": df[col].value_counts().head(max_unique_vals).to_dict()
            })
            
        catalog["columns"][col] = col_info
        
    return catalog

def catalog_to_json(catalog: dict) -> str:
    """LLM 프롬프트에 넣기 좋게 JSON 문자열로 변환합니다."""
    return json.dumps(catalog, indent=2, ensure_ascii=False)
