"""
RAG API Routes
문서 기반 질의응답 및 RAG 시스템 엔드포인트
"""

import os
from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from services.rag_service import get_rag_service
from config.settings import UPLOAD_DIR


router = APIRouter(prefix="/api/rag", tags=["RAG"])


# Request/Response 모델
class AskQuestionRequest(BaseModel):
    question: str
    file_uris: List[str]
    conversation_id: Optional[str] = None
    model_name: Optional[str] = "gemini-2.5-flash"


class ChatRequest(BaseModel):
    message: str
    file_uris: List[str]
    conversation_id: str
    model_name: Optional[str] = "gemini-2.5-flash"


class GenerateQuizRequest(BaseModel):
    file_uri: str
    num_questions: Optional[int] = 5
    difficulty: Optional[str] = "medium"
    model_name: Optional[str] = "gemini-2.5-flash"


# uploads 디렉토리가 없으면 생성
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload-and-index")
async def upload_and_index_document(
    file: UploadFile = File(...),
    knowledge_base_name: Optional[str] = Form("default")
):
    """
    문서 업로드 및 지식 베이스에 인덱싱

    Request:
    - file: 업로드할 문서 파일
    - knowledge_base_name: 지식 베이스 이름 (기본: "default")

    Returns:
    - success: 성공 여부
    - file_uri: 업로드된 파일 URI
    - message: 메시지
    """
    # PDF 파일 확인
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="현재 PDF 파일만 지원합니다."
        )

    # 파일 저장
    file_content = await file.read()
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(temp_file_path, "wb") as f:
            f.write(file_content)

        # RAG 서비스로 문서 추가
        rag_service = get_rag_service()

        # 지식 베이스가 없으면 생성
        try:
            rag_service.create_knowledge_base(
                name=knowledge_base_name,
                display_name=knowledge_base_name
            )
        except:
            # 이미 존재하면 무시
            pass

        # 문서 추가
        result = rag_service.add_document_to_knowledge_base(
            file_path=temp_file_path,
            knowledge_base_name=knowledge_base_name,
            display_name=file.filename
        )

        return {
            "success": True,
            "file_uri": result.get("file_name"),
            "display_name": result.get("display_name"),
            "knowledge_base": knowledge_base_name,
            "message": "문서 업로드 및 인덱싱 완료"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"문서 업로드 실패: {str(e)}"
        )

    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@router.post("/ask")
async def ask_question(request: AskQuestionRequest):
    """
    문서 기반 질문에 답변

    Request body:
    {
        "question": "질문 내용",
        "file_uris": ["files/xxx", "files/yyy"],
        "conversation_id": "optional_conversation_id",
        "model_name": "gemini-2.5-flash"
    }

    Returns:
    - answer: AI 답변
    - sources: 참조한 문서 목록
    - conversation_id: 대화 ID
    """
    try:
        rag_service = get_rag_service()

        result = rag_service.ask_question(
            question=request.question,
            file_uris=request.file_uris,
            conversation_id=request.conversation_id,
            model_name=request.model_name
        )

        return {
            "success": True,
            **result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"질문 답변 실패: {str(e)}"
        )


@router.post("/chat")
async def chat_with_documents(request: ChatRequest):
    """
    문서 기반 채팅 (대화 이력 유지)

    Request body:
    {
        "message": "메시지 내용",
        "file_uris": ["files/xxx", "files/yyy"],
        "conversation_id": "conversation_123",
        "model_name": "gemini-2.5-flash"
    }

    Returns:
    - answer: AI 답변
    - conversation_id: 대화 ID
    - history_length: 대화 이력 길이
    """
    try:
        rag_service = get_rag_service()

        result = rag_service.chat_with_documents(
            message=request.message,
            file_uris=request.file_uris,
            conversation_id=request.conversation_id,
            model_name=request.model_name
        )

        return {
            "success": True,
            **result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"채팅 실패: {str(e)}"
        )


@router.get("/conversation/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """
    대화 이력 조회

    Returns:
    - conversation_id: 대화 ID
    - history: 대화 이력 리스트
    """
    try:
        rag_service = get_rag_service()

        history = rag_service.get_conversation_history(conversation_id)

        return {
            "success": True,
            "conversation_id": conversation_id,
            "history": history,
            "message_count": len(history)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"대화 이력 조회 실패: {str(e)}"
        )


@router.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """
    대화 이력 삭제

    Returns:
    - success: 성공 여부
    - message: 메시지
    """
    try:
        rag_service = get_rag_service()

        rag_service.clear_conversation_history(conversation_id)

        return {
            "success": True,
            "message": f"대화 이력 '{conversation_id}' 삭제 완료"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"대화 이력 삭제 실패: {str(e)}"
        )


@router.post("/generate-quiz")
async def generate_quiz(request: GenerateQuizRequest):
    """
    문서 내용 기반 퀴즈 생성

    Request body:
    {
        "file_uri": "files/xxx",
        "num_questions": 5,
        "difficulty": "medium",
        "model_name": "gemini-2.5-flash"
    }

    Returns:
    - quiz: 생성된 퀴즈 데이터
    - source_file: 원본 문서 이름
    """
    try:
        rag_service = get_rag_service()

        result = rag_service.generate_quiz_from_document(
            file_uri=request.file_uri,
            num_questions=request.num_questions,
            difficulty=request.difficulty,
            model_name=request.model_name
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"퀴즈 생성 실패: {str(e)}"
        )


@router.post("/search")
async def search_knowledge_base(
    query: str = Form(...),
    knowledge_base_name: str = Form("default"),
    top_k: int = Form(5)
):
    """
    지식 베이스 검색

    Request:
    - query: 검색 쿼리
    - knowledge_base_name: 지식 베이스 이름
    - top_k: 반환할 결과 개수

    Returns:
    - results: 검색 결과 리스트
    - query: 검색 쿼리
    - result_count: 결과 개수
    """
    try:
        rag_service = get_rag_service()

        results = rag_service.search_knowledge_base(
            query=query,
            knowledge_base_name=knowledge_base_name,
            top_k=top_k
        )

        return {
            "success": True,
            "query": query,
            "knowledge_base": knowledge_base_name,
            "results": results,
            "result_count": len(results)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"검색 실패: {str(e)}"
        )


@router.get("/knowledge-bases")
async def list_knowledge_bases():
    """
    모든 지식 베이스 목록 조회

    Returns:
    - knowledge_bases: 지식 베이스 목록
    - count: 개수
    """
    try:
        rag_service = get_rag_service()
        embedding_service = rag_service.embedding_service

        corpora = embedding_service.list_corpora()

        return {
            "success": True,
            "knowledge_bases": corpora,
            "count": len(corpora)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"지식 베이스 목록 조회 실패: {str(e)}"
        )


@router.post("/knowledge-bases")
async def create_knowledge_base(
    name: str = Form(...),
    display_name: Optional[str] = Form(None)
):
    """
    새로운 지식 베이스 생성

    Request:
    - name: 지식 베이스 이름 (고유)
    - display_name: 표시 이름 (선택)

    Returns:
    - name: 생성된 지식 베이스 이름
    - display_name: 표시 이름
    """
    try:
        rag_service = get_rag_service()

        result = rag_service.create_knowledge_base(
            name=name,
            display_name=display_name
        )

        return {
            "success": True,
            **result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"지식 베이스 생성 실패: {str(e)}"
        )


@router.delete("/knowledge-bases/{name}")
async def delete_knowledge_base(name: str):
    """
    지식 베이스 삭제

    Returns:
    - success: 성공 여부
    - message: 메시지
    """
    try:
        rag_service = get_rag_service()
        embedding_service = rag_service.embedding_service

        embedding_service.delete_corpus(name)

        return {
            "success": True,
            "message": f"지식 베이스 '{name}' 삭제 완료"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"지식 베이스 삭제 실패: {str(e)}"
        )
