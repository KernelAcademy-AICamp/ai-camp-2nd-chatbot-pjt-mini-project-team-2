"""
RAG Service - 고수준 RAG 기능 제공
EmbeddingService를 활용하여 문서 기반 질의응답 시스템 구현
"""

from typing import List, Dict, Optional, Any
import google.generativeai as genai
from google.generativeai.types import File
from fastapi import HTTPException

from services.embedding_service import get_embedding_service
from services.pdf_service import get_pdf_loader


class RAGService:
    """
    고수준 RAG(Retrieval-Augmented Generation) 서비스
    문서 업로드, 검색, 답변 생성을 통합 관리
    """

    def __init__(self):
        """RAGService 초기화"""
        self.embedding_service = get_embedding_service()
        self.pdf_loader = get_pdf_loader()
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}

    def create_knowledge_base(
        self,
        name: str,
        display_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        지식 베이스(Knowledge Base) 생성

        Args:
            name: 지식 베이스 이름
            display_name: 표시 이름

        Returns:
            생성된 지식 베이스 정보
        """
        try:
            result = self.embedding_service.create_corpus(
                corpus_name=name,
                display_name=display_name
            )

            print(f"✅ 지식 베이스 생성: {name}")

            return result

        except Exception as e:
            print(f"❌ 지식 베이스 생성 실패: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create knowledge base: {str(e)}"
            )

    def add_document_to_knowledge_base(
        self,
        file_path: str,
        knowledge_base_name: str,
        display_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        지식 베이스에 문서 추가

        Args:
            file_path: 문서 파일 경로
            knowledge_base_name: 지식 베이스 이름
            display_name: 문서 표시 이름

        Returns:
            추가된 문서 정보
        """
        try:
            result = self.embedding_service.upload_file_to_corpus(
                file_path=file_path,
                corpus_name=knowledge_base_name,
                display_name=display_name
            )

            print(f"✅ 문서 추가: {display_name or file_path}")

            return result

        except Exception as e:
            print(f"❌ 문서 추가 실패: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to add document: {str(e)}"
            )

    def search_knowledge_base(
        self,
        query: str,
        knowledge_base_name: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        지식 베이스 검색

        Args:
            query: 검색 쿼리
            knowledge_base_name: 검색할 지식 베이스 이름
            top_k: 반환할 결과 개수

        Returns:
            검색 결과 리스트
        """
        try:
            results = self.embedding_service.search_in_corpus(
                query=query,
                corpus_name=knowledge_base_name,
                top_k=top_k
            )

            return results

        except Exception as e:
            print(f"❌ 검색 실패: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to search knowledge base: {str(e)}"
            )

    def ask_question(
        self,
        question: str,
        file_uris: List[str],
        conversation_id: Optional[str] = None,
        model_name: str = "gemini-2.5-flash"
    ) -> Dict[str, Any]:
        """
        문서 기반 질문에 답변 (간단한 RAG)

        Args:
            question: 사용자 질문
            file_uris: 참조할 파일 URI 리스트 (files/xxx 형식)
            conversation_id: 대화 ID (대화 이력 관리용)
            model_name: 사용할 Gemini 모델

        Returns:
            답변 정보
        """
        try:
            # 파일 객체 가져오기
            files = []
            for uri in file_uris:
                file_obj = self.embedding_service.get_file_by_name(uri)
                if file_obj:
                    files.append(file_obj)
                else:
                    print(f"⚠️ 파일을 찾을 수 없습니다: {uri}")

            if not files:
                raise ValueError("유효한 파일이 없습니다.")

            # 대화 이력 가져오기
            history = []
            if conversation_id and conversation_id in self.conversation_history:
                history = self.conversation_history[conversation_id]

            # 답변 생성
            result = self.embedding_service.generate_answer_simple(
                query=question,
                files=files,
                model_name=model_name
            )

            # 대화 이력 저장
            if conversation_id:
                if conversation_id not in self.conversation_history:
                    self.conversation_history[conversation_id] = []

                self.conversation_history[conversation_id].append({
                    "role": "user",
                    "content": question
                })
                self.conversation_history[conversation_id].append({
                    "role": "assistant",
                    "content": result["answer"]
                })

            result["conversation_id"] = conversation_id

            return result

        except Exception as e:
            print(f"❌ 질문 답변 실패: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to answer question: {str(e)}"
            )

    def chat_with_documents(
        self,
        message: str,
        file_uris: List[str],
        conversation_id: str,
        model_name: str = "gemini-2.5-flash"
    ) -> Dict[str, Any]:
        """
        문서 기반 채팅 (대화 이력 유지)

        Args:
            message: 사용자 메시지
            file_uris: 참조할 파일 URI 리스트
            conversation_id: 대화 ID
            model_name: 사용할 Gemini 모델

        Returns:
            답변 정보
        """
        try:
            # 파일 객체 가져오기
            files = []
            for uri in file_uris:
                file_obj = self.embedding_service.get_file_by_name(uri)
                if file_obj:
                    files.append(file_obj)

            if not files:
                raise ValueError("유효한 파일이 없습니다.")

            # 대화 이력 가져오기
            if conversation_id not in self.conversation_history:
                self.conversation_history[conversation_id] = []

            history = self.conversation_history[conversation_id]

            # 모델 생성
            model = genai.GenerativeModel(model_name)

            # 대화 이력을 포함한 프롬프트 구성
            system_prompt = """당신은 업로드된 문서를 기반으로 답변하는 전문 AI 어시스턴트입니다.

**역할:**
- 문서의 내용을 정확하게 이해하고 답변합니다
- 이전 대화 맥락을 고려하여 일관된 답변을 제공합니다
- 문서에 없는 내용은 추측하지 않고 솔직히 말합니다
- 한국어로 친절하고 명확하게 답변합니다
"""

            # 대화 이력 포함
            conversation_text = "\n".join([
                f"{'사용자' if msg['role'] == 'user' else 'AI'}: {msg['content']}"
                for msg in history
            ])

            user_prompt = f"""**이전 대화:**
{conversation_text if conversation_text else '(첫 대화입니다)'}

**현재 질문:** {message}

위 문서들과 대화 맥락을 참고하여 답변해주세요.
"""

            # 답변 생성
            content = files + [system_prompt, user_prompt]
            response = model.generate_content(content)

            answer = response.text

            # 대화 이력 업데이트
            self.conversation_history[conversation_id].append({
                "role": "user",
                "content": message
            })
            self.conversation_history[conversation_id].append({
                "role": "assistant",
                "content": answer
            })

            print(f"💬 대화 {conversation_id}: {len(self.conversation_history[conversation_id])}개 메시지")

            return {
                "conversation_id": conversation_id,
                "message": message,
                "answer": answer,
                "sources": [f.display_name for f in files],
                "model": model_name,
                "history_length": len(self.conversation_history[conversation_id])
            }

        except Exception as e:
            print(f"❌ 채팅 실패: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to chat with documents: {str(e)}"
            )

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        대화 이력 조회

        Args:
            conversation_id: 대화 ID

        Returns:
            대화 이력 리스트
        """
        return self.conversation_history.get(conversation_id, [])

    def clear_conversation_history(self, conversation_id: str):
        """
        대화 이력 삭제

        Args:
            conversation_id: 대화 ID
        """
        if conversation_id in self.conversation_history:
            del self.conversation_history[conversation_id]
            print(f"🗑️ 대화 이력 삭제: {conversation_id}")

    def generate_quiz_from_document(
        self,
        file_uri: str,
        num_questions: int = 5,
        difficulty: str = "medium",
        model_name: str = "gemini-2.5-flash"
    ) -> Dict[str, Any]:
        """
        문서 내용 기반 퀴즈 생성

        Args:
            file_uri: 문서 파일 URI
            num_questions: 생성할 문제 수
            difficulty: 난이도 (easy, medium, hard)
            model_name: 사용할 Gemini 모델

        Returns:
            생성된 퀴즈 정보
        """
        try:
            # 파일 객체 가져오기
            file_obj = self.embedding_service.get_file_by_name(file_uri)
            if not file_obj:
                raise ValueError(f"파일을 찾을 수 없습니다: {file_uri}")

            # 모델 생성
            model = genai.GenerativeModel(model_name)

            # 퀴즈 생성 프롬프트
            difficulty_kr = {"easy": "쉬움", "medium": "보통", "hard": "어려움"}.get(difficulty, "보통")

            prompt = f"""이 문서의 내용을 기반으로 **{num_questions}개의 객관식 문제**를 생성해주세요.

**요구사항:**
- 난이도: {difficulty_kr}
- 각 문제는 4개의 선택지를 가져야 합니다
- 정답과 해설을 포함해주세요
- 문서의 핵심 내용을 다루는 문제여야 합니다

**출력 형식 (JSON):**
```json
{{
    "quiz_title": "퀴즈 제목",
    "total_questions": {num_questions},
    "difficulty": "{difficulty}",
    "questions": [
        {{
            "question_number": 1,
            "question_text": "문제 내용",
            "options": [
                "1) 선택지 1",
                "2) 선택지 2",
                "3) 선택지 3",
                "4) 선택지 4"
            ],
            "correct_answer": "정답 번호 (1~4)",
            "explanation": "해설 내용"
        }}
    ]
}}
```

JSON 형식으로만 응답해주세요.
"""

            # 퀴즈 생성
            response = model.generate_content([file_obj, prompt])

            # JSON 파싱
            import json
            result_text = response.text.strip()

            # 마크다운 코드 블록 제거
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            quiz_data = json.loads(result_text)

            print(f"✅ 퀴즈 생성 완료: {num_questions}개 문제")

            return {
                "success": True,
                "quiz": quiz_data,
                "source_file": file_obj.display_name
            }

        except Exception as e:
            print(f"❌ 퀴즈 생성 실패: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate quiz: {str(e)}"
            )


# Global RAGService instance (싱글톤 패턴)
_rag_service_instance: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """
    RAGService 싱글톤 인스턴스 반환

    Returns:
        RAGService 인스턴스
    """
    global _rag_service_instance

    if _rag_service_instance is None:
        _rag_service_instance = RAGService()

    return _rag_service_instance
