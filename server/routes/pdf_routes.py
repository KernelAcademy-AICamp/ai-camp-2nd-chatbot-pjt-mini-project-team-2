"""
PDF API Routes
PDF 파일 업로드 및 텍스트 추출 엔드포인트
"""

import os
import shutil
from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from typing import Dict, Any, List, Optional

from services.pdf_service import get_pdf_loader
from services.vector_db_service import get_vector_db_service
from config.settings import UPLOAD_DIR, MAX_FILE_SIZE


router = APIRouter(prefix="/api/pdf", tags=["PDF"])


# uploads 디렉토리가 없으면 생성
os.makedirs(UPLOAD_DIR, exist_ok=True)


async def save_uploaded_file_streaming(upload_file: UploadFile, dest_path: str) -> int:
    """
    업로드된 파일을 스트리밍 방식으로 저장 (메모리 효율적)

    Args:
        upload_file: FastAPI UploadFile 객체
        dest_path: 저장할 경로

    Returns:
        파일 크기 (바이트)

    Raises:
        HTTPException: 파일 크기가 MAX_FILE_SIZE를 초과하는 경우
    """
    file_size = 0
    try:
        with open(dest_path, "wb") as f:
            while chunk := await upload_file.read(8192):  # 8KB 청크
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    os.remove(dest_path)
                    raise HTTPException(
                        status_code=400,
                        detail=f"파일 크기가 너무 큽니다. 최대 {MAX_FILE_SIZE // (1024*1024)}MB까지 업로드 가능합니다."
                    )
                f.write(chunk)
        return file_size
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise HTTPException(
            status_code=500,
            detail=f"파일 저장 중 오류 발생: {str(e)}"
        )


@router.post("/upload")
async def upload_pdf_file(file: UploadFile = File(...)):
    """
    PDF 파일 업로드 및 Gemini API에 등록 (최적화됨)

    Request:
    - file: PDF 파일 (multipart/form-data)

    Returns:
    - success: 성공 여부
    - file_info: 업로드된 파일 정보
    - message: 메시지
    """
    # PDF 파일 확인
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="PDF 파일만 업로드 가능합니다."
        )

    # 파일을 임시로 저장
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # 파일 저장 (스트리밍 방식)
        await save_uploaded_file_streaming(file, temp_file_path)

        # PDFLoader로 Gemini API에 업로드
        loader = get_pdf_loader()
        uploaded_file = loader.upload_pdf(temp_file_path, display_name=file.filename)

        # 파일 처리 대기
        processed_file = loader.wait_for_file_processing(uploaded_file)

        # 파일 정보 반환
        file_info = loader.get_file_info(processed_file)

        # 🆕 FAISS 벡터 DB 생성 (LangChain 활용)
        vector_db_info = None
        try:
            vector_db_service = get_vector_db_service()

            # PDF에서 전체 텍스트 추출 (하이브리드 방식)
            full_text = loader.extract_full_text(processed_file, file_path=temp_file_path)

            # 벡터 스토어 생성
            vector_store = vector_db_service.create_vector_store_from_text(
                text=full_text,
                file_name=file.filename,
                metadata={
                    "gemini_file_uri": processed_file.uri,
                    "display_name": file.filename
                }
            )

            # 벡터 스토어 정보 조회
            vector_db_info = vector_db_service.get_store_info(file.filename)

            print(f"✅ FAISS 벡터 DB 생성 완료: {vector_db_info.get('total_chunks', 0)}개 청크")
        except Exception as e:
            print(f"⚠️ 벡터 DB 생성 실패 (무시): {str(e)}")
            vector_db_info = {"status": "failed", "error": str(e)}

        return {
            "success": True,
            "file_info": file_info,
            "vector_db_info": vector_db_info,
            "message": "PDF 파일 업로드, 처리 및 FAISS 벡터 DB 생성 완료"
        }

    except Exception as e:
        # 임시 파일 삭제
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        raise HTTPException(
            status_code=500,
            detail=f"파일 업로드 중 오류 발생: {str(e)}"
        )


@router.post("/extract-text")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    """
    PDF 파일에서 전체 텍스트 추출 (최적화됨)

    Request:
    - file: PDF 파일 (multipart/form-data)

    Returns:
    - success: 성공 여부
    - text: 추출된 텍스트
    - file_name: 파일 이름
    """
    # PDF 파일 확인
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="PDF 파일만 업로드 가능합니다."
        )

    # 파일을 임시로 저장 (스트리밍 방식)
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # 파일 저장 (스트리밍 방식)
        await save_uploaded_file_streaming(file, temp_file_path)

        # PDFLoader로 Gemini API에 업로드
        loader = get_pdf_loader()
        uploaded_file = loader.upload_pdf(temp_file_path, display_name=file.filename)

        # 파일 처리 대기
        processed_file = loader.wait_for_file_processing(uploaded_file)

        # 텍스트 추출 (하이브리드 방식: PyPDF2 우선, 실패 시 Gemini)
        extracted_text = loader.extract_full_text(processed_file, file_path=temp_file_path)

        # Gemini API에서 파일 삭제 (선택사항)
        # loader.delete_file(processed_file)

        return {
            "success": True,
            "text": extracted_text,
            "file_name": file.filename,
            "text_length": len(extracted_text)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"텍스트 추출 중 오류 발생: {str(e)}"
        )

    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@router.post("/extract-preview")
async def extract_preview_from_pdf(file: UploadFile = File(...)):
    """
    PDF 파일의 내용 미리보기 (요약)

    Request:
    - file: PDF 파일 (multipart/form-data)

    Returns:
    - success: 성공 여부
    - preview: 문서 요약
    - file_name: 파일 이름
    """
    # PDF 파일 확인
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="PDF 파일만 업로드 가능합니다."
        )

    # 파일을 임시로 저장
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # 파일 저장 (스트리밍 방식)
        await save_uploaded_file_streaming(file, temp_file_path)

        # PDFLoader로 Gemini API에 업로드
        loader = get_pdf_loader()
        uploaded_file = loader.upload_pdf(temp_file_path, display_name=file.filename)

        # 파일 처리 대기
        processed_file = loader.wait_for_file_processing(uploaded_file)

        # 미리보기 추출
        preview = loader.extract_text_preview(processed_file)

        return {
            "success": True,
            "preview": preview,
            "file_name": file.filename
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"미리보기 생성 중 오류 발생: {str(e)}"
        )

    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@router.post("/extract-structured")
async def extract_structured_content(file: UploadFile = File(...)):
    """
    PDF 파일에서 구조화된 콘텐츠 추출
    (자격증 시험 문제집 또는 교재에 특화)

    Request:
    - file: PDF 파일 (multipart/form-data)

    Returns:
    - success: 성공 여부
    - content: 구조화된 콘텐츠 (JSON)
    - file_name: 파일 이름
    """
    # PDF 파일 확인
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="PDF 파일만 업로드 가능합니다."
        )

    # 파일을 임시로 저장
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # 파일 저장 (스트리밍 방식)
        await save_uploaded_file_streaming(file, temp_file_path)

        # PDFLoader로 Gemini API에 업로드
        loader = get_pdf_loader()
        uploaded_file = loader.upload_pdf(temp_file_path, display_name=file.filename)

        # 파일 처리 대기
        processed_file = loader.wait_for_file_processing(uploaded_file)

        # 구조화된 콘텐츠 추출
        structured_content = loader.extract_structured_content(processed_file)

        return {
            "success": True,
            "content": structured_content,
            "file_name": file.filename
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"구조화된 콘텐츠 추출 중 오류 발생: {str(e)}"
        )

    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@router.post("/extract-by-pages")
async def extract_text_by_pages(file: UploadFile = File(...)):
    """
    PDF 파일의 텍스트를 페이지별로 추출

    Request:
    - file: PDF 파일 (multipart/form-data)

    Returns:
    - success: 성공 여부
    - pages: 페이지별 텍스트 리스트
    - file_name: 파일 이름
    - total_pages: 총 페이지 수
    """
    # PDF 파일 확인
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="PDF 파일만 업로드 가능합니다."
        )

    # 파일을 임시로 저장
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # 파일 저장 (스트리밍 방식)
        await save_uploaded_file_streaming(file, temp_file_path)

        # PDFLoader로 Gemini API에 업로드
        loader = get_pdf_loader()
        uploaded_file = loader.upload_pdf(temp_file_path, display_name=file.filename)

        # 파일 처리 대기
        processed_file = loader.wait_for_file_processing(uploaded_file)

        # 페이지별 텍스트 추출
        pages = loader.extract_text_by_pages(processed_file)

        return {
            "success": True,
            "pages": pages,
            "file_name": file.filename,
            "total_pages": len(pages)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"페이지별 추출 중 오류 발생: {str(e)}"
        )

    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@router.get("/uploaded-files")
async def list_uploaded_files():
    """
    현재 세션에서 업로드된 파일 목록 조회

    Returns:
    - files: 업로드된 파일 목록
    - count: 파일 개수
    """
    try:
        loader = get_pdf_loader()
        files = loader.list_uploaded_files()

        return {
            "success": True,
            "files": files,
            "count": len(files)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"파일 목록 조회 중 오류 발생: {str(e)}"
        )


@router.delete("/delete-file/{file_name:path}")
async def delete_single_file(file_name: str):
    """
    Gemini API에서 특정 파일 삭제

    Args:
    - file_name: 삭제할 파일의 name (예: files/abc123)

    Returns:
    - success: 성공 여부
    - message: 메시지
    """
    try:
        loader = get_pdf_loader()

        # 파일 찾기
        file_to_delete = None
        for file in loader.uploaded_files:
            if file.name == file_name:
                file_to_delete = file
                break

        if file_to_delete is None:
            raise HTTPException(
                status_code=404,
                detail=f"파일을 찾을 수 없습니다: {file_name}"
            )

        # 파일 삭제
        loader.delete_file(file_to_delete)

        return {
            "success": True,
            "message": f"파일이 삭제되었습니다: {file_to_delete.display_name}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"파일 삭제 중 오류 발생: {str(e)}"
        )


@router.delete("/clear-files")
async def clear_all_files():
    """
    Gemini API에 업로드된 모든 파일 삭제

    Returns:
    - success: 성공 여부
    - message: 메시지
    """
    try:
        loader = get_pdf_loader()
        loader.delete_all_files()

        return {
            "success": True,
            "message": "모든 파일이 삭제되었습니다."
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"파일 삭제 중 오류 발생: {str(e)}"
        )
