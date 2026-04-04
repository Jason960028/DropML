import pandas as pd
import numpy as np
from profiler import generate_data_catalog, catalog_to_json

def run_test():
    print("📊 1. 가상의 테스트 데이터를 생성합니다...")
    np.random.seed(42)
    num_rows = 150000 # 🌟 10만 개 이상 생성하여 샘플링 기능 테스트
    
    # 숫자형, 문자열(범주형), 날짜형 컬럼 생성
    df = pd.DataFrame({
        'id': np.arange(num_rows),
        'age': np.random.randint(18, 80, size=num_rows).astype(float), # 결측치를 넣기 위해 float로
        'income': np.random.normal(50000, 15000, size=num_rows),
        'gender': np.random.choice(['Male', 'Female', 'Unknown'], size=num_rows, p=[0.45, 0.45, 0.10]),
        'signup_date': pd.date_range(start='2020-01-01', periods=num_rows, freq='min')
    })

    # ✔️ 'age'와 'income' 컬럼에 강제로 결측치(NaN) 추가
    missing_age_indices = np.random.choice(df.index, size=5000, replace=False)
    missing_income_indices = np.random.choice(df.index, size=15000, replace=False)
    df.loc[missing_age_indices, 'age'] = np.nan
    df.loc[missing_income_indices, 'income'] = np.nan

    print(f"✅ 원본 데이터 Shape: {df.shape}")
    print("-" * 50)
    
    print("⚙️ 2. profiler.py 엔진을 실행합니다...")
    # 프로파일링 실행 (최대 고유값: 3, 10만건 샘플링 캡)
    catalog = generate_data_catalog(df, max_unique_vals=3, max_sample_rows=100000)
    
    print("-" * 50)
    print("🚀 3. 결과 (Data Catalog JSON 포맷):")
    
    # 결과를 JSON 문자열로 변환하여 출력
    json_result = catalog_to_json(catalog)
    print(json_result)
    
if __name__ == "__main__":
    run_test()
