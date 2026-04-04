import pandas as pd
import numpy as np

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    발생 가능한 다양한 Edge Case를 사전에 방지하기 위해 
    데이터프레임을 글로벌하게 전처리(Data Cleaning)합니다.
    """
    # 1. 컬럼명 앞뒤 공백 제거 (사용자/AI가 컬럼명 타이핑 시 발생하는 Key 에러 방지)
    if isinstance(df.columns, pd.Index) and df.columns.dtype == 'object':
        df.columns = df.columns.astype(str).str.strip()

    # 2. 문자열(object) 컬럼들의 앞뒤 공백 제거 (가장 흔한 ZeroDivision 및 필터링 실패 원인)
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        # 문자열 타입인 데이터만 공백을 자릅니다 (NaN 값 유지)
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # 3. 흔하게 쓰이는 비표준 결측치 표현들을 파이썬 표준 np.nan으로 강제 통일
    null_variants = ["?", "-", "NA", "na", "N/A", "n/a", "null", "Null", "None", "none", "#DIV/0!", "#N/A"]
    df.replace(null_variants, np.nan, inplace=True)

    # 4. 스페이스바만 쳐져 있는(빈 공백 문자열) 값들도 np.nan으로 변환
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # 5. 완전 비어있는 (모든 값이 NaN인) 찌꺼기 행/열 제거
    df.dropna(how='all', axis=0, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)

    # 6. 문자열로 잘못 인식된 숫자형 컬럼 자동 형변환 안전 시도
    for col in obj_cols:
        try:
            df_converted = pd.to_numeric(df[col], errors='ignore')
            df[col] = df_converted
        except Exception:
            pass

    # 7. 남아있는 NaN 값을 안전하게 채우기
    #    - 숫자 컬럼: 중앙값(median)으로 채움 (mean보다 이상치에 강건)
    #    - 문자열 컬럼: 'Unknown'으로 채움
    #    → 이 단계 덕분에 AI가 df.fillna(df.mean()) 같은 위험한 코드를 쓸 필요가 없어짐
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in str_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna('Unknown')

    return df
