저녁에 바로 작업을 시작하실 수 있도록, **Reproducing CatDB** 프로젝트의 **Design Document**를 CLI(Command Line Interface) 가이드 형식으로 정리해 드립니다. 이 가이드를 따라가시면 프로젝트 생성부터 배포까지 막힘없이 진행하실 수 있습니다.

---

# 📑 Design Document: Reproducing CatDB
**Author:** Woosuk Chang  
**Tech Stack:** Python, Streamlit, Pandas, Gemini API, Ollama (Llama 3)  
[cite_start]**Goal:** 데이터 카탈로그 기반의 지능형 ML 파이프라인 생성기 구축 [cite: 1, 3]

---

## 🛠 Phase 1: Environment Setup
가장 먼저 개발 환경을 구축합니다. 터미널(CLI)에서 아래 명령어를 순서대로 실행하세요.

### **1. 프로젝트 폴더 생성 및 이동**
```bash
mkdir catdb-repro && cd catdb-repro
```

### **2. 가상환경 설정 (추천)**
```bash
python -m venv venv
# Windows:
source venv/Scripts/activate
# Mac/Linux:
source venv/bin/activate
```

### **3. 필수 라이브러리 설치**
```bash
pip install streamlit pandas google-generativeai requests
```

---

## 🧠 Phase 2: Core Engine - Data Profiling
[cite_start]사용자가 업로드한 데이터의 '설명서(Catalog)'를 뽑아내는 핵심 모듈입니다. [cite: 5]

### **핵심 로직 (profiler.py)**
* **Schema Extraction**: 컬럼명과 데이터 타입($int, float, object$) 구분.
* **Statistics**: 평균($\mu$), 표준편차($\sigma$), 사분위수 계산.
* [cite_start]**Missing Value Analysis**: 컬럼별 결측치 비율($\%$) 산출. [cite: 5]

---

## 🤖 Phase 3: LLM Integration (Hybrid)
[cite_start]상황에 따라 Gemini와 Llama 3를 선택하여 사용합니다. [cite: 13, 14]

### **1. Gemini API (Performance Mode)**
* [cite_start]`google-generativeai` 라이브러리를 사용하여 원격 서버와 통신합니다. [cite: 7]
* 복잡한 논리 구조와 최신 ML 라이브러리 활용 능력이 뛰어납니다.

### **2. Ollama / Llama 3 (Privacy Mode)**
* [cite_start]로컬 서버(`http://localhost:11434/api/generate`)에 JSON 요청을 보냅니다. [cite: 14]
* [cite_start]데이터 보안이 중요한 메타데이터 분석에 활용합니다. [cite: 14]

---

## 🎨 Phase 4: UI Development (Streamlit)
사용자가 체감하는 UX를 구현합니다.

### **Step-by-Step Flow**
1.  **`st.file_uploader`**: 사용자가 CSV 파일을 업로드합니다.
2.  **`st.dataframe`**: 업로드된 데이터의 미리보기를 출력합니다.
3.  [cite_start]**`st.selectbox`**: Gemini와 Llama 3 중 하나를 선택합니다. [cite: 13, 14]
4.  **`st.text_input`**: 분석 목표(Goal)를 입력받습니다. (예: "Predict Titanic survival") [cite_start][cite: 16]

---

## 🔄 Phase 5: Self-Healing & XAI
[cite_start]생성된 코드가 완벽하게 돌아가도록 보장하고 이유를 설명합니다. [cite: 18, 21]

### **1. Error Management Loop**
* `exec()` 함수를 사용해 생성된 코드를 실행해 봅니다.
* [cite_start]에러 발생 시 `traceback`을 LLM에게 다시 전달하여 수정을 요청합니다. [cite: 21]

### **2. Explainable AI (XAI)**
* [cite_start]LLM에게 "왜 이 전처리 기법을 선택했는지"를 데이터 카탈로그의 통계치를 근거로 설명하게 합니다. [cite: 18]

---

## 🚀 Phase 6: Deployment Strategy
[cite_start]프로젝트를 완성한 후 배포하는 단계입니다. [cite: 23]

### **1. GitHub Repository 구성**
* `app.py`: 메인 실행 파일.
* `requirements.txt`: 라이브러리 목록.
* `README.md`: 프로젝트 설명 및 설치 가이드 (Ollama 설치법 포함).

### **2. Streamlit Community Cloud (Demo)**
* GitHub 저장소를 연결하여 웹 주소 생성.
* 환경 변수(Secrets)에 `GOOGLE_API_KEY` 등록.

---