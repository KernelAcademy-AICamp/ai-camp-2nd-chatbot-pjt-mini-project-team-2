"""
Vector Database Service - LangChain + FAISS
PDF 문서를 청킹하고 임베딩하여 FAISS 벡터 DB에 저장
"""

import os
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from config.settings import OPENAI_API_KEY, UPLOAD_DIR


class VectorDBService:
    """
    LangChain + FAISS를 활용한 벡터 데이터베이스 서비스

    주요 기능:
    - PDF 텍스트 청킹 (RecursiveCharacterTextSplitter)
    - OpenAI Embeddings API를 통한 임베딩 생성 (text-embedding-3-small)
    - FAISS 벡터 스토어 생성 및 관리
    - 시맨틱 검색
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        VectorDBService 초기화

        Args:
            api_key: OpenAI API 키 (None이면 설정에서 가져옴)
        """
        self.api_key = api_key or OPENAI_API_KEY

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for embeddings")

        # OpenAI Embeddings 초기화 (text-embedding-3-small 모델)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.api_key
        )

        # 텍스트 스플리터 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 청크 크기 (토큰 수)
            chunk_overlap=200,  # 청크 간 오버랩
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # 벡터 스토어 디렉토리
        self.vector_store_dir = os.path.join(UPLOAD_DIR, "vector_stores")
        os.makedirs(self.vector_store_dir, exist_ok=True)

        # 현재 로드된 벡터 스토어들 (파일명 -> FAISS 객체)
        self.vector_stores: Dict[str, FAISS] = {}

    def create_vector_store_from_text(
        self,
        text: str,
        file_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FAISS:
        """
        텍스트로부터 벡터 스토어 생성

        Args:
            text: 전체 텍스트
            file_name: 파일 이름 (벡터 스토어 식별용)
            metadata: 메타데이터 (선택사항)

        Returns:
            생성된 FAISS 벡터 스토어
        """
        print(f"📚 벡터 스토어 생성 시작: {file_name}")

        # 1. 텍스트 청킹
        print(f"   ✂️ 텍스트 청킹 중...")
        chunks = self.text_splitter.split_text(text)
        print(f"   ✅ {len(chunks)}개 청크 생성 완료")

        # 2. Document 객체 생성 (메타데이터 포함)
        documents = []
        for idx, chunk in enumerate(chunks):
            doc_metadata = {
                "source": file_name,
                "chunk_index": idx,
                "total_chunks": len(chunks)
            }
            if metadata:
                doc_metadata.update(metadata)

            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))

        # 3. FAISS 벡터 스토어 생성
        print(f"   🔢 임베딩 생성 및 FAISS 인덱스 구축 중...")
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        print(f"   ✅ 벡터 스토어 생성 완료")

        # 4. 메모리에 저장
        self.vector_stores[file_name] = vector_store

        # 5. 디스크에 저장
        self.save_vector_store(file_name, vector_store)

        return vector_store

    def save_vector_store(self, file_name: str, vector_store: FAISS) -> str:
        """
        벡터 스토어를 디스크에 저장

        Args:
            file_name: 파일 이름
            vector_store: FAISS 벡터 스토어

        Returns:
            저장된 경로
        """
        # 파일명에서 확장자 제거 및 안전한 경로 생성
        safe_name = Path(file_name).stem
        store_path = os.path.join(self.vector_store_dir, safe_name)

        # FAISS 저장 (자동으로 .faiss와 .pkl 파일 생성)
        vector_store.save_local(store_path)

        print(f"💾 벡터 스토어 저장 완료: {store_path}")
        return store_path

    def load_vector_store(self, file_name: str) -> Optional[FAISS]:
        """
        디스크에서 벡터 스토어 로드

        Args:
            file_name: 파일 이름

        Returns:
            로드된 FAISS 벡터 스토어 (없으면 None)
        """
        # 이미 메모리에 있으면 반환
        if file_name in self.vector_stores:
            print(f"📦 벡터 스토어 캐시에서 로드: {file_name}")
            return self.vector_stores[file_name]

        # 디스크에서 로드
        safe_name = Path(file_name).stem
        store_path = os.path.join(self.vector_store_dir, safe_name)

        if not os.path.exists(store_path):
            print(f"⚠️ 벡터 스토어를 찾을 수 없음: {store_path}")
            return None

        try:
            print(f"📂 벡터 스토어 로드 중: {store_path}")
            vector_store = FAISS.load_local(
                store_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True  # pickle 역직렬화 허용
            )

            # 메모리에 캐시
            self.vector_stores[file_name] = vector_store
            print(f"✅ 벡터 스토어 로드 완료")

            return vector_store

        except Exception as e:
            print(f"❌ 벡터 스토어 로드 실패: {str(e)}")
            return None

    def search(
        self,
        query: str,
        file_name: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        벡터 스토어에서 시맨틱 검색

        Args:
            query: 검색 쿼리
            file_name: 검색할 파일 이름
            top_k: 반환할 최대 결과 수

        Returns:
            검색 결과 리스트 (각 결과는 {'text', 'metadata', 'score'} 포함)
        """
        # 벡터 스토어 로드
        vector_store = self.load_vector_store(file_name)

        if vector_store is None:
            print(f"⚠️ 벡터 스토어가 없어 검색 불가: {file_name}")
            return []

        print(f"🔍 시맨틱 검색 시작: '{query}' (top_k={top_k})")

        # 유사도 검색 (점수 포함)
        results = vector_store.similarity_search_with_score(
            query=query,
            k=top_k
        )

        # 결과 포맷팅
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)  # FAISS는 거리(낮을수록 유사)를 반환
            })

        print(f"✅ {len(formatted_results)}개 결과 찾음")

        return formatted_results

    def delete_vector_store(self, file_name: str) -> bool:
        """
        벡터 스토어 삭제 (메모리 및 디스크)

        Args:
            file_name: 파일 이름

        Returns:
            삭제 성공 여부
        """
        try:
            # 메모리에서 삭제
            if file_name in self.vector_stores:
                del self.vector_stores[file_name]

            # 디스크에서 삭제
            safe_name = Path(file_name).stem
            store_path = os.path.join(self.vector_store_dir, safe_name)

            if os.path.exists(store_path):
                import shutil
                shutil.rmtree(store_path)
                print(f"🗑️ 벡터 스토어 삭제 완료: {file_name}")
                return True

            return False

        except Exception as e:
            print(f"❌ 벡터 스토어 삭제 실패: {str(e)}")
            return False

    def list_vector_stores(self) -> List[str]:
        """
        저장된 벡터 스토어 목록 조회

        Returns:
            벡터 스토어 이름 리스트
        """
        try:
            if not os.path.exists(self.vector_store_dir):
                return []

            stores = [
                d for d in os.listdir(self.vector_store_dir)
                if os.path.isdir(os.path.join(self.vector_store_dir, d))
            ]

            return stores

        except Exception as e:
            print(f"❌ 벡터 스토어 목록 조회 실패: {str(e)}")
            return []

    def get_store_info(self, file_name: str) -> Optional[Dict[str, Any]]:
        """
        벡터 스토어 정보 조회

        Args:
            file_name: 파일 이름

        Returns:
            벡터 스토어 정보 (총 청크 수 등)
        """
        vector_store = self.load_vector_store(file_name)

        if vector_store is None:
            return None

        # FAISS 인덱스에서 벡터 개수 확인
        total_vectors = vector_store.index.ntotal

        return {
            "file_name": file_name,
            "total_chunks": total_vectors,
            "embedding_dimension": vector_store.index.d
        }


# Global VectorDBService instance (싱글톤 패턴)
_vector_db_service_instance: Optional[VectorDBService] = None


def get_vector_db_service() -> VectorDBService:
    """
    VectorDBService 싱글톤 인스턴스 반환

    Returns:
        VectorDBService 인스턴스
    """
    global _vector_db_service_instance

    if _vector_db_service_instance is None:
        _vector_db_service_instance = VectorDBService()

    return _vector_db_service_instance
