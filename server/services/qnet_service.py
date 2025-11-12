"""
Q-Net API Service
Handles all Q-Net API requests
"""

import httpx
import asyncio
import time
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlencode
from fastapi import HTTPException

from config.settings import QNET_SERVICE_KEY

# 서버 사이드 메모리 캐시
_cache: Dict[str, Tuple[float, str]] = {}
CACHE_TTL = 7 * 24 * 60 * 60  # 7일 (초 단위)


async def make_qnet_request(base_url: str, endpoint: str, params: Dict[str, Any], max_retries: int = 3) -> tuple:
    """
    Make HTTP request to Q-Net API with retry logic and server-side caching

    Args:
        base_url: Base URL of the Q-Net API
        endpoint: API endpoint
        params: Query parameters
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (status_code, response_text)
    """
    # Build query params like Node.js URLSearchParams
    query_params = {
        "serviceKey": QNET_SERVICE_KEY,
        **params
    }

    # Build query string using urlencode
    query_string = urlencode(query_params)
    url = f"{base_url}/{endpoint}?{query_string}"

    # 캐시 키 생성 (URL 전체를 키로 사용)
    cache_key = f"{endpoint}:{query_string}"

    # 캐시 확인
    if cache_key in _cache:
        cached_time, cached_data = _cache[cache_key]
        if time.time() - cached_time < CACHE_TTL:
            print(f"✅ 서버 캐시에서 즉시 반환: {endpoint} (캐시 나이: {int((time.time() - cached_time) / 60)}분)")
            return 200, cached_data
        else:
            # 만료된 캐시 삭제
            del _cache[cache_key]
            print(f"🗑️ 만료된 캐시 삭제: {cache_key}")

    print(f"🔗 Q-Net API 요청 중: {url}")

    last_error = None

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                response = await client.get(url)
                print(f"✅ Q-Net API Response (attempt {attempt + 1}): status={response.status_code}, length={len(response.text)}")

                # Check for API errors in XML
                if "resultCode" in response.text:
                    if "<resultCode>99</resultCode>" in response.text:
                        error_msg = "Q-Net API returned error code 99"
                        if "resultMsg" in response.text:
                            import re
                            msg_match = re.search(r'<resultMsg>(.*?)</resultMsg>', response.text)
                            if msg_match:
                                error_msg = f"Q-Net API Error: {msg_match.group(1)}"
                        print(f"⚠️ {error_msg}")
                        print(f"⚠️ Response preview: {response.text[:500]}")

                        # Retry for error code 99
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 2  # Exponential backoff
                            print(f"🔄 Retrying in {wait_time} seconds...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise HTTPException(status_code=503, detail=error_msg)

                # 성공 시 캐시에 저장
                _cache[cache_key] = (time.time(), response.text)
                print(f"💾 서버 캐시에 저장: {cache_key}")

                return response.status_code, response.text

        except httpx.TimeoutException as e:
            last_error = f"Request timeout: {str(e)}"
            print(f"❌ Timeout Error (attempt {attempt + 1}/{max_retries}): {last_error}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"🔄 Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            continue

        except httpx.RequestError as e:
            last_error = f"Request error: {str(e)}"
            print(f"❌ Request Error (attempt {attempt + 1}/{max_retries}): {last_error}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"🔄 Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            continue

        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            print(f"❌ Unexpected Error (attempt {attempt + 1}/{max_retries}): {last_error}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"🔄 Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            continue

    # All retries failed
    print(f"❌ All {max_retries} attempts failed. Last error: {last_error}")
    raise HTTPException(status_code=500, detail=last_error or "Q-Net API request failed after multiple retries")
