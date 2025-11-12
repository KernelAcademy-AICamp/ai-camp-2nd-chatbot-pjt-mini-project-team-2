"""
PDF 파일 로더 및 파싱 모듈
Gemini API의 File API를 활용하여 PDF 파일을 업로드하고 처리합니다.
"""

import os
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import google.generativeai as genai
from google.generativeai.types import File
from fastapi import HTTPException

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

from config.settings import GEMINI_API_KEY


class PDFLoader:
    """PDF 파일을 Gemini API에 업로드하고 관리하는 클래스"""

    def __init__(self, api_key: Optional[str] = None):
        """
        PDFLoader 초기화

        Args:
            api_key: Google AI Studio에서 발급받은 API 키 (None이면 설정에서 가져옴)
        """
        self.api_key = api_key or GEMINI_API_KEY

        if not self.api_key:
            raise HTTPException(
                status_code=503,
                detail="Gemini API key is not configured. Please set GEMINI_API_KEY in .env file"
            )

        genai.configure(api_key=self.api_key)
        self.uploaded_files: List[File] = []
        self.file_cache: Dict[str, File] = {}  # 파일 해시 -> File 객체 캐시

    def _get_file_hash(self, file_path: str) -> str:
        """
        파일의 MD5 해시값 계산

        Args:
            file_path: 파일 경로

        Returns:
            MD5 해시 문자열
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            # 대용량 파일을 위해 청크 단위로 읽기
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def upload_pdf(self, file_path: str, display_name: Optional[str] = None, use_cache: bool = True) -> File:
        """
        PDF 파일을 Gemini API에 업로드 (캐싱 지원)

        Args:
            file_path: 업로드할 PDF 파일 경로
            display_name: 파일의 표시 이름 (선택사항)
            use_cache: 캐시 사용 여부 (기본: True)

        Returns:
            업로드된 File 객체

        Raises:
            FileNotFoundError: 파일이 존재하지 않을 경우
            ValueError: PDF 파일이 아닌 경우
        """
        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        # PDF 파일 확인
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("PDF 파일만 업로드 가능합니다.")

        # 캐시 확인
        if use_cache:
            file_hash = self._get_file_hash(file_path)
            if file_hash in self.file_cache:
                cached_file = self.file_cache[file_hash]
                # 캐시된 파일이 여전히 유효한지 확인
                try:
                    file_status = genai.get_file(cached_file.name)
                    if file_status.state.name == "ACTIVE":
                        print(f"♻️ 캐시된 파일 사용: {cached_file.display_name}")
                        return cached_file
                except:
                    # 캐시된 파일이 무효화된 경우 제거
                    del self.file_cache[file_hash]

        # 표시 이름 설정
        if display_name is None:
            display_name = Path(file_path).stem

        print(f"📤 PDF 파일 업로드 중: {file_path}")

        # Gemini API에 파일 업로드
        uploaded_file = genai.upload_file(
            path=file_path,
            display_name=display_name
        )

        print(f"✅ 업로드 완료: {uploaded_file.display_name}")
        print(f"   - URI: {uploaded_file.uri}")
        print(f"   - MIME Type: {uploaded_file.mime_type}")

        # 업로드된 파일 목록에 추가
        self.uploaded_files.append(uploaded_file)

        # 캐시에 저장
        if use_cache:
            self.file_cache[file_hash] = uploaded_file

        return uploaded_file

    def wait_for_file_processing(self, uploaded_file: File, timeout: int = 300) -> File:
        """
        파일 처리가 완료될 때까지 대기 (Exponential Backoff 적용)

        Args:
            uploaded_file: 업로드된 File 객체
            timeout: 최대 대기 시간 (초)

        Returns:
            처리 완료된 File 객체

        Raises:
            TimeoutError: 처리 시간이 timeout을 초과한 경우
            RuntimeError: 파일 처리 중 오류 발생 시
        """
        print(f"⏳ 파일 처리 대기 중: {uploaded_file.display_name}")

        start_time = time.time()
        wait_time = 0.5  # 초기 대기 시간을 0.5초로 단축

        while True:
            # 파일 상태 확인
            file_status = genai.get_file(uploaded_file.name)

            # 처리 완료
            if file_status.state.name == "ACTIVE":
                elapsed = time.time() - start_time
                print(f"✅ 파일 처리 완료: {file_status.display_name} ({elapsed:.1f}초)")
                return file_status

            # 처리 실패
            elif file_status.state.name == "FAILED":
                raise RuntimeError(f"파일 처리 실패: {file_status.display_name}")

            # 타임아웃 체크
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                raise TimeoutError(
                    f"파일 처리 시간 초과 ({timeout}초): {uploaded_file.display_name}"
                )

            # Exponential backoff으로 대기 (최대 3초)
            time.sleep(wait_time)
            wait_time = min(wait_time * 1.5, 3.0)
            print(f"   처리 중... (경과 시간: {int(elapsed_time)}초)")

    def upload_multiple_pdfs(self, file_paths: List[str]) -> List[File]:
        """
        여러 PDF 파일을 한 번에 업로드

        Args:
            file_paths: 업로드할 PDF 파일 경로 리스트

        Returns:
            업로드된 File 객체 리스트
        """
        uploaded_files = []

        for file_path in file_paths:
            try:
                uploaded_file = self.upload_pdf(file_path)
                processed_file = self.wait_for_file_processing(uploaded_file)
                uploaded_files.append(processed_file)
            except Exception as e:
                print(f"❌ 파일 업로드 실패: {file_path}")
                print(f"   오류: {str(e)}")
                continue

        return uploaded_files

    def get_file_info(self, file: File) -> Dict:
        """
        업로드된 파일의 정보 조회

        Args:
            file: File 객체

        Returns:
            파일 정보 딕셔너리
        """
        return {
            "name": file.name,
            "display_name": file.display_name,
            "uri": file.uri,
            "mime_type": file.mime_type,
            "size_bytes": getattr(file, 'size_bytes', None),
            "state": file.state.name,
            "create_time": str(getattr(file, 'create_time', None)),
        }

    def list_uploaded_files(self) -> List[Dict]:
        """
        업로드된 모든 파일의 정보 조회

        Returns:
            파일 정보 딕셔너리 리스트
        """
        return [self.get_file_info(file) for file in self.uploaded_files]

    def delete_file(self, file: File):
        """
        업로드된 파일 삭제

        Args:
            file: 삭제할 File 객체
        """
        try:
            genai.delete_file(file.name)
            print(f"🗑️ 파일 삭제 완료: {file.display_name}")

            # 목록에서 제거
            if file in self.uploaded_files:
                self.uploaded_files.remove(file)
        except Exception as e:
            print(f"❌ 파일 삭제 실패: {file.display_name}")
            print(f"   오류: {str(e)}")

    def delete_all_files(self):
        """업로드된 모든 파일 삭제"""
        for file in self.uploaded_files[:]:  # 복사본으로 순회
            self.delete_file(file)

    def extract_text_with_pypdf2(self, file_path: str) -> Tuple[bool, str]:
        """
        PyPDF2를 사용한 빠른 텍스트 추출 (텍스트 기반 PDF에 적합)

        Args:
            file_path: PDF 파일 경로

        Returns:
            (성공 여부, 추출된 텍스트) 튜플
        """
        if not PYPDF2_AVAILABLE:
            return False, ""

        try:
            reader = PdfReader(file_path)
            text_parts = []

            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    text_parts.append(f"=== 페이지 {page_num} ===\n{text}\n")

            full_text = "\n".join(text_parts)

            # 충분한 텍스트가 추출되었는지 확인 (최소 100자)
            if len(full_text.strip()) > 100:
                print(f"⚡ PyPDF2로 빠른 텍스트 추출 완료 ({len(full_text)} 문자)")
                return True, full_text
            else:
                print(f"⚠️ PyPDF2 추출 결과 부족 ({len(full_text)} 문자), Gemini로 전환")
                return False, ""

        except Exception as e:
            print(f"⚠️ PyPDF2 추출 실패: {str(e)}, Gemini로 전환")
            return False, ""

    def extract_full_text(self, file: File, model_name: str = "gemini-2.5-flash", file_path: Optional[str] = None, use_fast_extraction: bool = True) -> str:
        """
        PDF 파일의 전체 텍스트 추출 (하이브리드 방식)

        이 메서드는 성능 최적화를 위해:
        1. 먼저 PyPDF2로 빠른 추출 시도 (텍스트 기반 PDF)
        2. 실패 시 Gemini의 멀티모달 기능 활용 (스캔/이미지 PDF, OCR 필요)

        Args:
            file: Gemini File 객체
            model_name: 사용할 Gemini 모델 (기본: gemini-2.5-flash)
            file_path: 로컬 파일 경로 (PyPDF2 사용을 위해 필요)
            use_fast_extraction: PyPDF2 빠른 추출 사용 여부

        Returns:
            추출된 전체 텍스트
        """
        # 1. 빠른 추출 시도 (PyPDF2)
        if use_fast_extraction and file_path and PYPDF2_AVAILABLE:
            success, text = self.extract_text_with_pypdf2(file_path)
            if success:
                return text

        # 2. Gemini를 사용한 고급 추출 (OCR, 이미지 PDF 지원)
        print(f"🤖 Gemini AI로 고급 텍스트 추출 중...")
        model = genai.GenerativeModel(model_name)

        prompt = """
        이 PDF 문서의 모든 텍스트를 빠짐없이 추출해주세요.

        **중요 지침:**
        1. 문서의 모든 텍스트를 순서대로 추출하세요
        2. 이미지나 스캔된 부분이 있다면 OCR로 텍스트를 인식하세요
        3. 표나 목록은 구조를 유지하면서 텍스트로 변환하세요
        4. 페이지 번호나 헤더/푸터도 포함하세요
        5. 요약하지 말고 원문 그대로 추출하세요
        6. 수식이나 특수 문자도 가능한 한 정확히 표현하세요

        추출한 텍스트만 출력하고, 다른 설명은 추가하지 마세요.
        """

        try:
            response = model.generate_content([file, prompt])
            return response.text
        except Exception as e:
            return f"텍스트 추출 실패: {str(e)}"

    def extract_text_by_pages(self, file: File, model_name: str = "gemini-2.5-flash") -> List[Dict[str, str]]:
        """
        PDF 파일의 텍스트를 페이지별로 추출

        Args:
            file: File 객체
            model_name: 사용할 Gemini 모델

        Returns:
            페이지별 텍스트 딕셔너리 리스트 [{"page": 1, "text": "..."}, ...]
        """
        model = genai.GenerativeModel(model_name)

        prompt = """
        이 PDF 문서의 텍스트를 페이지별로 추출해주세요.

        **출력 형식 (JSON):**
        ```json
        {
            "pages": [
                {"page": 1, "text": "1페이지의 모든 텍스트..."},
                {"page": 2, "text": "2페이지의 모든 텍스트..."},
                ...
            ]
        }
        ```

        **지침:**
        - 각 페이지의 모든 텍스트를 빠짐없이 추출
        - 이미지/스캔 부분은 OCR 적용
        - 표와 목록의 구조 유지
        - JSON 형식으로만 응답
        """

        try:
            response = model.generate_content([file, prompt])
            import json
            # JSON 파싱 시도
            result_text = response.text.strip()
            # 마크다운 코드 블록 제거
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            data = json.loads(result_text)
            return data.get("pages", [])
        except Exception as e:
            print(f"페이지별 추출 실패: {str(e)}")
            # 실패 시 전체 텍스트를 단일 페이지로 반환
            full_text = self.extract_full_text(file, model_name)
            return [{"page": 1, "text": full_text}]

    def extract_text_preview(self, file: File, model_name: str = "gemini-2.5-flash") -> str:
        """
        PDF 파일의 텍스트 내용 미리보기 추출 (요약)

        Args:
            file: File 객체
            model_name: 사용할 Gemini 모델

        Returns:
            추출된 텍스트 미리보기
        """
        model = genai.GenerativeModel(model_name)

        prompt = """
        이 PDF 문서의 내용을 요약해주세요.
        다음 정보를 포함해주세요:
        1. 문서의 주요 주제
        2. 목차 구조 (있는 경우)
        3. 주요 내용 개요
        4. 문서의 총 페이지 수 (추정)
        5. 문서 유형 (텍스트 기반 / 이미지/스캔 기반)

        한국어로 답변해주세요.
        """

        try:
            response = model.generate_content([file, prompt])
            return response.text
        except Exception as e:
            return f"텍스트 추출 실패: {str(e)}"

    def extract_structured_content(self, file: File, model_name: str = "gemini-2.5-flash") -> Dict:
        """
        PDF에서 구조화된 콘텐츠 추출 (제목, 본문, 문제, 보기 등)
        자격증 시험 문제집에 특화된 추출

        Args:
            file: File 객체
            model_name: 사용할 Gemini 모델

        Returns:
            구조화된 콘텐츠 딕셔너리
        """
        model = genai.GenerativeModel(model_name)

        prompt = """
        이 문서를 분석하여 다음 형식의 JSON으로 변환해주세요.

        **문서가 자격증 시험 문제집인 경우:**
        ```json
        {
            "document_type": "exam",
            "title": "문서 제목",
            "questions": [
                {
                    "question_number": 1,
                    "question_text": "문제 내용",
                    "options": ["1번 선택지", "2번 선택지", "3번 선택지", "4번 선택지"],
                    "answer": "정답 번호 또는 텍스트",
                    "explanation": "해설 (있는 경우)"
                }
            ]
        }
        ```

        **문서가 개념서/교재인 경우:**
        ```json
        {
            "document_type": "textbook",
            "title": "문서 제목",
            "chapters": [
                {
                    "chapter_number": 1,
                    "chapter_title": "챕터 제목",
                    "sections": [
                        {
                            "section_title": "섹션 제목",
                            "content": "본문 내용"
                        }
                    ]
                }
            ]
        }
        ```

        문서 유형을 판단하고 적절한 형식으로 추출하세요.
        JSON 형식으로만 응답하세요.
        """

        try:
            response = model.generate_content([file, prompt])
            import json
            result_text = response.text.strip()

            # 마크다운 코드 블록 제거
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            return json.loads(result_text)
        except Exception as e:
            return {
                "error": f"구조화 추출 실패: {str(e)}",
                "document_type": "unknown"
            }


# Global PDFLoader instance (싱글톤 패턴)
_pdf_loader_instance: Optional[PDFLoader] = None


def get_pdf_loader() -> PDFLoader:
    """
    PDFLoader 싱글톤 인스턴스 반환

    Returns:
        PDFLoader 인스턴스
    """
    global _pdf_loader_instance

    if _pdf_loader_instance is None:
        _pdf_loader_instance = PDFLoader()

    return _pdf_loader_instance
