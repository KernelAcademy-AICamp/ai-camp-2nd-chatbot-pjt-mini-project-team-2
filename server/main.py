"""
FastAPI Server for Q-Net API Proxy and OpenAI Integration
Handles CORS, proxies requests to Q-Net OpenAPI, and provides AI-powered study planning
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import QNET_TEST_INFO_API, QNET_QUALIFICATION_API, OPENAI_API_KEY, GEMINI_API_KEY
from routes.qnet_routes import router as qnet_router
from routes.openai_routes import router as openai_router
from routes.pdf_routes import router as pdf_router
from routes.rag_routes import router as rag_router
from routes.pdfupload_routes import router as quiz_router

# Initialize FastAPI app
app = FastAPI(title="Q-Pass API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(qnet_router)
app.include_router(openai_router)
app.include_router(pdf_router)
app.include_router(rag_router)
app.include_router(quiz_router)


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "services": {
            "testInfo": QNET_TEST_INFO_API,
            "qualification": QNET_QUALIFICATION_API,
            "openai": "enabled" if OPENAI_API_KEY else "disabled",
            "gemini": "enabled" if GEMINI_API_KEY else "disabled"
        }
    }


if __name__ == "__main__":
    import uvicorn

    print("""
╔════════════════════════════════════════════════════════╗
║            Q-Pass API Server Running                   ║
╚════════════════════════════════════════════════════════╝

  Port: 3001

  Services:
  - Q-Net API Proxy (시험 일정, 종목 목록)
  - OpenAI Integration (AI 학습 계획 생성)
  - Gemini PDF Parser (PDF 파일 파싱 및 텍스트 추출)
  - RAG System (문서 기반 질의응답 시스템)

  Available endpoints:
  ✅ GET  /api/health

  📅 Q-Net 시험 일정 조회:
  - GET /api/qnet/pe-list         (기술사)
  - GET /api/qnet/mc-list         (기능장)
  - GET /api/qnet/e-list          (기사, 산업기사)
  - GET /api/qnet/c-list          (기능사)
  - GET /api/qnet/fee-list        (종목별 수수료)
  - GET /api/qnet/jm-list         (종목별 일정)

  📋 Q-Net 국가기술자격:
  - GET /api/qnet/qualification-list

  🤖 OpenAI:
  - POST /api/openai/generate-study-plan  (학습 계획 생성)
  - POST /api/openai/chat                 (AI 챗봇)
  - POST /api/openai/chat-with-file       (파일 포함 챗봇)

  📄 PDF Processing (Gemini):
  - POST   /api/pdf/upload              (PDF 업로드)
  - POST   /api/pdf/extract-text        (전체 텍스트 추출)
  - POST   /api/pdf/extract-preview     (문서 요약)
  - POST   /api/pdf/extract-structured  (구조화된 콘텐츠 추출)
  - POST   /api/pdf/extract-by-pages    (페이지별 추출)
  - GET    /api/pdf/uploaded-files      (업로드 파일 목록)
  - DELETE /api/pdf/clear-files         (업로드 파일 삭제)

  🧠 RAG System (Gemini FileSearchTool):
  - POST   /api/rag/upload-and-index    (문서 업로드 및 인덱싱)
  - POST   /api/rag/ask                 (문서 기반 질문)
  - POST   /api/rag/chat                (문서 기반 채팅)
  - POST   /api/rag/generate-quiz       (문서 기반 퀴즈 생성)
  - POST   /api/rag/search              (지식 베이스 검색)
  - GET    /api/rag/knowledge-bases     (지식 베이스 목록)
  - POST   /api/rag/knowledge-bases     (지식 베이스 생성)
  - DELETE /api/rag/knowledge-bases/{name}  (지식 베이스 삭제)
  - GET    /api/rag/conversation/{id}   (대화 이력 조회)
  - DELETE /api/rag/conversation/{id}   (대화 이력 삭제)

  📝 Quiz Generation (Gemini AI):
  - POST   /api/quiz/upload-and-generate  (PDF 업로드 및 AI 퀴즈 생성)
  - GET    /api/quiz/health               (서비스 상태 확인)

  Ready to serve! 🚀
  """)

    uvicorn.run(app, host="0.0.0.0", port=3001)
