# main.py
import os
import random
import fitz  # PyMuPDF
import re
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
import nest_asyncio
import uvicorn

# ============================================
# ⚙️ FastAPI 앱 초기화
# ============================================
app = FastAPI(title="멀티에이전트 문제 생성 서버")

# ✅ CORS 설정 (React 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용 - 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================
# 🧩 PDF 업로드 엔드포인트
# ============================================
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # ✅ PDF 텍스트 추출
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text("text")

    # ✅ 텍스트 정제
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "", text)
    text = re.sub(r"페이지\s*\d+\s*/\s*\d+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[^가-힣a-zA-Z0-9.,?!\s]", "", text)
    text = re.sub(r"시나공|만든이|기출문제|자격증|블로그|카페", "", text, flags=re.IGNORECASE)

    # ✅ 문장 분리 및 문제 생성
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]

    def generate_mcq(sentence):
        words = sentence.split()
        if len(words) < 5:
            return None
        answer = random.choice(words).strip(",.!?():")
        options = [answer]
        while len(options) < 4:
            fake = random.choice(words).strip(",.!?():")
            if fake not in options:
                options.append(fake)
        random.shuffle(options)
        return {
            "question": sentence.replace(answer, "_____"),
            "options": options,
            "answer": answer,
        }

    quizzes = [q for s in sentences[:5] if (q := generate_mcq(s))]

    return {"status": "success", "questions": quizzes}

# ============================================
# 🧠 루트 엔드포인트
# ============================================
@app.get("/")
def root():
    return {"message": "멀티에이전트 서버 정상 작동 중 🚀"}

# ============================================
# 🚀 메인 실행부
# ============================================
if __name__ == "__main__":
    import nest_asyncio

    # ngrok 토큰 등록
    ngrok.set_auth_token("3594V0xG8PgXKGAGW4fxS4V6RzR_3oWjxFrJ6WMaW2jwysG44")

    # ngrok 포트 연결 (8000)
    public_url = ngrok.connect(8000)
    print("🌍 외부 접속 URL:", public_url.public_url)

    # Colab 환경이 아니더라도 nest_asyncio 적용 가능
    nest_asyncio.apply()

    # FastAPI 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)
