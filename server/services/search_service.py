"""
Search Service - Tavily API를 활용한 웹 검색
문서에 충분한 정보가 없을 때 웹 검색을 통해 정보 보충
"""

from typing import List, Dict, Optional, Any
from tavily import TavilyClient
from fastapi import HTTPException

from config.settings import TAVILY_API_KEY


class SearchService:
    """
    Tavily API를 활용한 웹 검색 서비스

    주요 기능:
    - 웹 검색 수행
    - 검색 결과 포맷팅
    - 관련성 높은 결과 필터링
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        SearchService 초기화

        Args:
            api_key: Tavily API 키 (None이면 설정에서 가져옴)
        """
        self.api_key = api_key or TAVILY_API_KEY

        if not self.api_key:
            raise HTTPException(
                status_code=503,
                detail="Tavily API key is not configured. Please set TAVILY_API_KEY in .env file"
            )

        self.client = TavilyClient(api_key=self.api_key)

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        웹 검색 수행

        Args:
            query: 검색 쿼리
            max_results: 최대 결과 개수 (기본: 5)
            search_depth: 검색 깊이 ("basic" 또는 "advanced", 기본: "advanced")
            include_domains: 포함할 도메인 리스트 (선택사항)
            exclude_domains: 제외할 도메인 리스트 (선택사항)

        Returns:
            검색 결과 딕셔너리
        """
        try:
            print(f"🔍 Tavily 검색 시작: '{query}'")
            print(f"   검색 깊이: {search_depth}, 최대 결과: {max_results}")

            # Tavily 검색 실행
            search_params = {
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
            }

            if include_domains:
                search_params["include_domains"] = include_domains

            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains

            response = self.client.search(**search_params)

            # 결과 개수 출력
            results_count = len(response.get("results", []))
            print(f"✅ 검색 완료: {results_count}개 결과")

            return response

        except Exception as e:
            print(f"❌ 검색 실패: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to perform web search: {str(e)}"
            )

    def search_and_format(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced"
    ) -> str:
        """
        웹 검색 수행 후 결과를 텍스트로 포맷팅

        Args:
            query: 검색 쿼리
            max_results: 최대 결과 개수
            search_depth: 검색 깊이

        Returns:
            포맷팅된 검색 결과 텍스트
        """
        try:
            # 검색 수행
            response = self.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth
            )

            # 결과 포맷팅
            results = response.get("results", [])

            if not results:
                return "검색 결과가 없습니다."

            formatted_text = f"**웹 검색 결과: '{query}'**\n\n"

            for idx, result in enumerate(results, 1):
                title = result.get("title", "제목 없음")
                url = result.get("url", "")
                content = result.get("content", "")
                score = result.get("score", 0)

                formatted_text += f"**[{idx}] {title}**\n"
                formatted_text += f"출처: {url}\n"
                formatted_text += f"관련도: {score:.2f}\n"
                formatted_text += f"{content}\n\n"

            return formatted_text

        except Exception as e:
            print(f"❌ 검색 결과 포맷팅 실패: {str(e)}")
            return f"검색 중 오류가 발생했습니다: {str(e)}"

    def get_answer_with_sources(
        self,
        query: str,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        검색 결과를 기반으로 답변 및 출처 반환

        Args:
            query: 검색 쿼리
            max_results: 최대 결과 개수

        Returns:
            답변과 출처 정보
        """
        try:
            # 검색 수행 (Tavily의 answer 기능 활용)
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=True  # 요약된 답변 포함
            )

            # 답변 추출
            answer = response.get("answer", "")
            results = response.get("results", [])

            # 출처 정보 추출
            sources = []
            for result in results:
                sources.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0)
                })

            return {
                "answer": answer,
                "sources": sources,
                "total_results": len(results)
            }

        except Exception as e:
            print(f"❌ 답변 생성 실패: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get answer with sources: {str(e)}"
            )


# Global SearchService instance (싱글톤 패턴)
_search_service_instance: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """
    SearchService 싱글톤 인스턴스 반환

    Returns:
        SearchService 인스턴스
    """
    global _search_service_instance

    if _search_service_instance is None:
        _search_service_instance = SearchService()

    return _search_service_instance
