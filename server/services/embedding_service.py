"""
Embedding Service - RAG 시스템을 위한 임베딩 및 벡터 검색
Google Gemini의 File Search Tool을 활용하여 자동 청킹, 임베딩, 벡터 검색 구현
"""

import os
import time
from typing import List, Dict, Optional, Any
import google.generativeai as genai
from google.generativeai.types import File
from fastapi import HTTPException

from config.settings import GEMINI_API_KEY


class EmbeddingService:
    """
    Google Gemini의 File Search Tool을 활용한 RAG 서비스

    FileSearchTool은 자동으로:
    1. 파일 청킹 (Chunking)
    2. 임베딩 생성 (Embeddings)
    3. 벡터 인덱싱 (Vector Indexing)
    4. 유사도 검색 (Similarity Search)
    을 처리합니다.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        EmbeddingService 초기화

        Args:
            api_key: Gemini API 키 (None이면 설정에서 가져옴)
        """
        self.api_key = api_key or GEMINI_API_KEY

        if not self.api_key:
            raise HTTPException(
                status_code=503,
                detail="Gemini API key is not configured. Please set GEMINI_API_KEY in .env file"
            )

        genai.configure(api_key=self.api_key)
        self.uploaded_files: List[File] = []
        self.vector_stores: Dict[str, Any] = {}  # corpus_name -> corpus_resource

    def create_corpus(self, corpus_name: str, display_name: Optional[str] = None) -> Dict[str, str]:
        """
        새로운 코퍼스(말뭉치) 생성
        코퍼스는 여러 문서를 그룹화하여 검색할 수 있는 컨테이너입니다.

        Args:
            corpus_name: 코퍼스의 고유 이름
            display_name: 표시 이름 (선택사항)

        Returns:
            생성된 코퍼스 정보
        """
        try:
            if display_name is None:
                display_name = corpus_name

            # 코퍼스 생성 (Semantic Retriever)
            corpus = genai.create_corpus(
                name=corpus_name,
                display_name=display_name
            )

            self.vector_stores[corpus_name] = corpus

            print(f"✅ 코퍼스 생성 완료: {corpus.name}")

            return {
                "name": corpus.name,
                "display_name": corpus.display_name,
                "status": "created"
            }

        except Exception as e:
            print(f"❌ 코퍼스 생성 실패: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create corpus: {str(e)}"
            )

    def upload_file_to_corpus(
        self,
        file_path: str,
        corpus_name: str,
        display_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        파일을 코퍼스에 업로드하고 자동 임베딩 생성

        Args:
            file_path: 업로드할 파일 경로
            corpus_name: 대상 코퍼스 이름
            display_name: 파일 표시 이름

        Returns:
            업로드된 파일 정보
        """
        try:
            # 파일 존재 확인
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

            # 표시 이름 설정
            if display_name is None:
                from pathlib import Path
                display_name = Path(file_path).stem

            print(f"📤 파일 업로드 중: {file_path}")
            print(f"   코퍼스: {corpus_name}")

            # 파일 업로드 (일반 업로드)
            uploaded_file = genai.upload_file(
                path=file_path,
                display_name=display_name
            )

            print(f"✅ 업로드 완료: {uploaded_file.display_name}")
            print(f"   URI: {uploaded_file.uri}")

            # 파일 처리 대기
            print("⏳ 파일 처리 대기 중...")
            start_time = time.time()
            timeout = 300  # 5분

            while uploaded_file.state.name != "ACTIVE":
                if time.time() - start_time > timeout:
                    raise TimeoutError("파일 처리 시간 초과")

                time.sleep(2)
                uploaded_file = genai.get_file(uploaded_file.name)

                if uploaded_file.state.name == "FAILED":
                    raise RuntimeError("파일 처리 실패")

            print(f"✅ 파일 처리 완료")

            # 코퍼스에 문서 생성 (임베딩 자동 생성)
            if corpus_name in self.vector_stores:
                corpus = self.vector_stores[corpus_name]

                # 문서를 코퍼스에 추가
                document = genai.create_document(
                    corpus_name=corpus.name,
                    display_name=display_name,
                    # 파일을 문서의 일부로 추가
                )

                print(f"✅ 코퍼스에 문서 추가 완료")
                print(f"   문서: {document.name}")

            # 업로드된 파일 목록에 추가
            self.uploaded_files.append(uploaded_file)

            return {
                "file_name": uploaded_file.name,
                "display_name": uploaded_file.display_name,
                "uri": uploaded_file.uri,
                "mime_type": uploaded_file.mime_type,
                "state": uploaded_file.state.name,
                "corpus": corpus_name,
                "status": "ready_for_search"
            }

        except Exception as e:
            print(f"❌ 파일 업로드 실패: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload file to corpus: {str(e)}"
            )

    def search_in_corpus(
        self,
        query: str,
        corpus_name: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        코퍼스에서 유사도 검색 수행

        Args:
            query: 검색 쿼리
            corpus_name: 검색할 코퍼스 이름
            top_k: 반환할 결과 개수

        Returns:
            검색 결과 리스트
        """
        try:
            if corpus_name not in self.vector_stores:
                raise ValueError(f"코퍼스를 찾을 수 없습니다: {corpus_name}")

            corpus = self.vector_stores[corpus_name]

            print(f"🔍 검색 중: '{query}'")
            print(f"   코퍼스: {corpus_name}")

            # 유사도 검색
            results = genai.query_corpus(
                corpus_name=corpus.name,
                query=query,
                results_count=top_k
            )

            # 결과 파싱
            search_results = []
            for idx, result in enumerate(results.relevant_chunks):
                search_results.append({
                    "rank": idx + 1,
                    "chunk_text": result.chunk_text,
                    "relevance_score": getattr(result, 'relevance_score', None),
                    "document_name": getattr(result, 'document_name', None),
                    "chunk_id": getattr(result, 'chunk_id', None)
                })

            print(f"✅ 검색 완료: {len(search_results)}개 결과")

            return search_results

        except Exception as e:
            print(f"❌ 검색 실패: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to search in corpus: {str(e)}"
            )

    def generate_answer_with_retrieval(
        self,
        query: str,
        files: List[File],
        model_name: str = "gemini-2.5-flash"
    ) -> Dict[str, Any]:
        """
        FileSearchTool을 사용하여 RAG 기반 답변 생성
        파일에서 관련 정보를 검색하고 답변을 생성합니다.

        Args:
            query: 사용자 질문
            files: 검색할 파일 리스트
            model_name: 사용할 Gemini 모델

        Returns:
            답변 및 검색 결과
        """
        try:
            print(f"💬 질문: {query}")
            print(f"📚 파일 수: {len(files)}개")

            # FileSearchTool 활성화하여 모델 생성
            model = genai.GenerativeModel(
                model_name=model_name,
                tools=[genai.protos.Tool(
                    google_search_retrieval=genai.protos.GoogleSearchRetrieval()
                )]
            )

            # 프롬프트 구성
            prompt = f"""다음 문서들을 참고하여 질문에 답변해주세요.

**질문:** {query}

**답변 지침:**
1. 업로드된 문서의 내용을 기반으로 정확하게 답변하세요
2. 문서에서 관련된 부분을 찾아 인용하세요
3. 답변에 확신이 없다면 솔직하게 말하세요
4. 한국어로 친절하고 명확하게 답변하세요

**답변:**
"""

            # 파일과 함께 질문
            response = model.generate_content([prompt] + files)

            answer = response.text

            print(f"✅ 답변 생성 완료")

            return {
                "query": query,
                "answer": answer,
                "sources": [f.display_name for f in files],
                "model": model_name
            }

        except Exception as e:
            print(f"❌ 답변 생성 실패: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate answer: {str(e)}"
            )

    def generate_answer_simple(
        self,
        query: str,
        files: List[File],
        model_name: str = "gemini-2.5-flash"
    ) -> Dict[str, Any]:
        """
        간단한 RAG: 파일 컨텍스트와 함께 질문
        (FileSearchTool 없이 직접 파일 전달)

        Args:
            query: 사용자 질문
            files: 검색할 파일 리스트
            model_name: 사용할 Gemini 모델

        Returns:
            답변 정보
        """
        try:
            print(f"💬 질문: {query}")
            print(f"📚 파일 수: {len(files)}개")

            # 모델 생성 (FileSearchTool 없음)
            model = genai.GenerativeModel(model_name)

            # 프롬프트 구성
            system_prompt = """당신은 업로드된 문서를 분석하여 정확한 답변을 제공하는 전문 AI 어시스턴트입니다.

**역할:**
- 업로드된 문서의 내용을 기반으로 질문에 답변합니다
- 문서에서 관련된 내용을 찾아 정확하게 인용합니다
- 답변에 확신이 없으면 솔직하게 말합니다
- 한국어로 친절하고 명확하게 답변합니다

**답변 형식:**
1. 핵심 답변을 먼저 제시
2. 문서에서 관련 내용을 인용하여 근거 제시
3. 필요시 추가 설명이나 예시 제공
"""

            user_prompt = f"""**질문:** {query}

위 문서들을 참고하여 질문에 답변해주세요.
"""

            # 파일과 함께 질문
            content = files + [user_prompt]
            response = model.generate_content(content)

            answer = response.text

            print(f"✅ 답변 생성 완료")

            return {
                "query": query,
                "answer": answer,
                "sources": [f.display_name for f in files],
                "model": model_name,
                "method": "simple_rag"
            }

        except Exception as e:
            print(f"❌ 답변 생성 실패: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate answer: {str(e)}"
            )

    def list_corpora(self) -> List[Dict[str, str]]:
        """
        생성된 모든 코퍼스 목록 조회

        Returns:
            코퍼스 정보 리스트
        """
        try:
            corpora = genai.list_corpora()

            corpus_list = []
            for corpus in corpora:
                corpus_list.append({
                    "name": corpus.name,
                    "display_name": corpus.display_name,
                    "create_time": str(getattr(corpus, 'create_time', None)),
                })

            return corpus_list

        except Exception as e:
            print(f"❌ 코퍼스 목록 조회 실패: {str(e)}")
            return []

    def delete_corpus(self, corpus_name: str):
        """
        코퍼스 삭제

        Args:
            corpus_name: 삭제할 코퍼스 이름
        """
        try:
            genai.delete_corpus(name=corpus_name)

            if corpus_name in self.vector_stores:
                del self.vector_stores[corpus_name]

            print(f"🗑️ 코퍼스 삭제 완료: {corpus_name}")

        except Exception as e:
            print(f"❌ 코퍼스 삭제 실패: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete corpus: {str(e)}"
            )

    def get_file_by_name(self, file_name: str) -> Optional[File]:
        """
        파일 이름으로 File 객체 조회

        Args:
            file_name: 파일 이름 (files/xxx 형식)

        Returns:
            File 객체 또는 None
        """
        try:
            return genai.get_file(file_name)
        except Exception as e:
            print(f"❌ 파일 조회 실패: {str(e)}")
            return None


# Global EmbeddingService instance (싱글톤 패턴)
_embedding_service_instance: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    EmbeddingService 싱글톤 인스턴스 반환

    Returns:
        EmbeddingService 인스턴스
    """
    global _embedding_service_instance

    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()

    return _embedding_service_instance
