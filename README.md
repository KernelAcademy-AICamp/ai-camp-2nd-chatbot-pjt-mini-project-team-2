# **LLM API를 활용한 학습 보조 챗봇 개발**
본 프로젝트는 OpenAI, Claude 등 최신 대규모 언어 모델(LLM) API를 활용하여 사용자의 학습 효율을 높이는 보조 챗봇 프로토타입을 개발합니다. 사용자가 제공한 문서(강의 노트, 기사 등)를 기반으로 **요약, Q&A, 문제 생성, 관련 자료 추천** 등 핵심적인 학습 지원 기능을 제공하는 것을 목표로 합니다.

- **프로젝트 기간:** 2025.10.16 ~ 2025.10.22 (총 1주일)
- **배포 링크:** [서비스 바로가기](링크 입력) *(완료 후 추가)*

---

## **1. 서비스 구성 요소**
### **1.1 주요 기능**
- **요약 및 Q&A 자동 생성 (1순위):** 사용자가 입력한 텍스트를 지정된 분량으로 요약하고, 해당 내용을 기반으로 예상 질문과 답변을 생성합니다.
- **맞춤형 과제/시험 문제 생성 (2순위):** 문서 내용을 기반으로 객관식, 주관식 등 다양한 형태의 문제를 자동 생성하여 사용자의 이해도 점검을 돕습니다.
- **학습 자료 추천 (3순위):** 문서의 핵심 키워드를 분석하여 심화 학습에 도움이 될 만한 관련 자료(기사, 블로그, 논문 등)를 추천합니다.

### **1.2 사용자 흐름**
- **사용자 시나리오 예시:**
  1. 사용자가 학습할 문서(텍스트 파일 또는 직접 입력)를 웹 UI에 업로드합니다.
  2. 챗봇이 문서의 핵심 내용을 5줄로 요약하여 보여줍니다.
  3. 사용자는 'Q&A 생성' 버튼을 클릭하여 문서 기반의 질의응답 세트를 확인합니다.
  4. 사용자는 '문제 생성' 버튼을 누르고 '객관식' 유형을 선택하여 3개의 문제를 받습니다.
  5. 사용자는 '관련 자료 추천' 버튼을 눌러 문서의 주제와 관련된 추가 학습 자료 링크를 확인합니다.

---

## **2. 활용 장비 및 협업 툴**

### **2.1 활용 장비**
- **개발 환경:** Windows 11 / macOS / Linux 기반 개인 PC
- **서버 환경:** Local 환경 구동 (필요시 Docker 컨테이너 활용)

### **2.2 협업 툴**
- **소스 관리:** GitHub
- **프로젝트 관리:** Notion, Jira
- **커뮤니케이션:** Slack, Discord
- **버전 관리:** Git

---

## **3. 최종 선정 AI 모델 구조**
- **모델 이름:** **OpenAI GPT-4o**, **Anthropic Claude 3** 등 (프로젝트에서 1종 이상 선택 활용)
- **구조 및 설명:** 본 프로젝트는 사전 학습된 LLM을 API 형태로 호출하여 사용합니다. 모델을 직접 학습하는 대신, **프롬프트 엔지니어링**을 통해 각 기능(요약, Q&A, 문제 생성)에 최적화된 결과물을 얻도록 제어하는 데 중점을 둡니다.
- **학습 데이터:** 사용자가 직접 입력하는 강의 노트, 기사, PDF 텍스트 등 비정형 데이터가 AI 모델의 주요 입력값으로 활용됩니다.
- **평가 지표:** 기능 요구사항 충족 여부를 기준으로 하며, 생성된 결과물(요약, 질문, 문제 등)의 **정확성, 일관성, 유용성**을 정성적으로 평가합니다.

---

## **4. 서비스 아키텍처**
### **4.1 시스템 구조도**
사용자 인터페이스(Streamlit)에서 입력을 받아 백엔드 서버(FastAPI)로 전달하고, 서버는 외부 LLM API와 통신하여 결과를 다시 사용자에게 보여주는 간단한 3-Tier 아키텍처를 따릅니다.
```
+------------------+      +---------------------+      +-----------------+
|   User (Client)  | <--> |   Backend Server    | <--> |  External LLM   |
| (Streamlit/Web)  |      | (FastAPI / Python)  |      | (OpenAI/Claude) |
+------------------+      +---------------------+      +-----------------+
```

### **4.2 데이터 흐름도**
1.  **사용자 입력:** 사용자가 UI를 통해 텍스트 문서를 입력합니다.
2.  **백엔드 요청:** Frontend(Streamlit)는 입력된 텍스트를 Backend(FastAPI) API로 전송합니다.
3.  **프롬프트 구성:** 백엔드는 사전에 설계된 프롬프트 템플릿에 사용자 텍스트를 결합합니다.
4.  **LLM API 호출:** 완성된 프롬프트를 OpenAI 또는 Claude API로 전송합니다.
5.  **결과 수신 및 파싱:** LLM이 생성한 응답(JSON, Bullet 형식)을 수신하여 파싱합니다.
6.  **결과 반환:** 파싱된 데이터를 UI가 표현하기 좋은 형태로 가공하여 Frontend로 반환합니다.

---

## **5. 사용 기술 스택**
### **5.1 백엔드**
- **Framework:** FastAPI (Python)
- **LLM API:** OpenAI, Anthropic

### **5.2 프론트엔드**
- **Framework:** Streamlit

### **5.3 머신러닝 및 데이터 분석**
- **LLM Libraries:** `openai`, `anthropic`
- **Data Handling:** `pandas` (필요시)

### **5.4 배포 및 운영**
- **Runtime Environment:** Python 3.9+
- **Containerization:** Docker (선택 사항)

---

## **6. 팀원 소개**


| ![박커널](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이커널](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최커널](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김커널](https://avatars.githubusercontent.com/u/156163982?v=4) | 
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | 
|            [박커널](https://github.com/)             |            [이커널](https://github.com/)             |            [최커널](https://github.com/)             |            [김커널](https://github.com/)             |   

---

## **7. Appendix**
### **7.1 참고 자료**
- **API 문서:** [OpenAI API Reference](https://platform.openai.com/docs/api-reference), [Anthropic API Documentation](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- **프레임워크:** [FastAPI 공식 문서](https://fastapi.tiangolo.com/), [Streamlit 공식 문서](https://docs.streamlit.io/)
- **프롬프트 가이드:** [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)

### **7.2 설치 및 실행 방법**
1.  **Repository 클론:**
    ```bash
    git clone [https://github.com/your-repo/your-project.git](https://github.com/your-repo/your-project.git)
    cd your-project
    ```

2.  **가상환경 생성 및 활성화:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3.  **필수 라이브러리 설치:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **.env 파일 생성 및 API 키 설정:**
    - 프로젝트 루트에 `.env` 파일을 생성하고 아래 내용을 추가합니다.
    ```
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"
    ```

5.  **애플리케이션 실행:**
    ```bash
    streamlit run app.py
    ```

### **7.3 주요 커밋 기록 및 업데이트 내역**

| 날짜 | 업데이트 내용 | 담당자 |
| :--- | :--- | :--- |
| YYYY.MM.DD | 프로젝트 초기 설정 및 `README.md` 작성 | 팀원1 |
| YYYY.MM.DD | 요약 및 Q&A 기능 프롬프트 설계 및 API 연동 | 팀원2 |
| YYYY.MM.DD | Streamlit 기본 UI 레이아웃 구현 | 팀원3 |
| YYYY.MM.DD | 문제 생성 기능 개발 및 출력 포맷 통일 | 팀원2 |
| YYYY.MM.DD | 전체 기능 통합 및 최종 테스트 | 전원 |
