"""
Q-Net API Routes
All endpoints for Q-Net API proxy
"""

from fastapi import APIRouter, Query
from fastapi.responses import Response
from typing import Optional

from services.qnet_service import make_qnet_request
from config.settings import QNET_TEST_INFO_API, QNET_QUALIFICATION_API

router = APIRouter(prefix="/api/qnet", tags=["Q-Net API"])


# ============================================
# 시험 일정 조회 APIs (InquiryTestInformationNTQSVC)
# ============================================

@router.get("/pe-list")
async def get_pe_list(implYy: Optional[str] = None, implSeq: Optional[str] = None):
    """기술사 시험 시행일정 조회"""
    params = {}
    if implYy:
        params["implYy"] = implYy
    if implSeq:
        params["implSeq"] = implSeq

    status_code, data = await make_qnet_request(QNET_TEST_INFO_API, "getPEList", params)
    return Response(content=data, media_type="application/xml", status_code=status_code)


@router.get("/mc-list")
async def get_mc_list(implYy: Optional[str] = None, implSeq: Optional[str] = None):
    """기능장 시험 시행일정 조회"""
    params = {}
    if implYy:
        params["implYy"] = implYy
    if implSeq:
        params["implSeq"] = implSeq

    status_code, data = await make_qnet_request(QNET_TEST_INFO_API, "getMCList", params)
    return Response(content=data, media_type="application/xml", status_code=status_code)


@router.get("/e-list")
async def get_e_list(implYy: Optional[str] = None, implSeq: Optional[str] = None):
    """기사, 산업기사 시험 시행일정 조회"""
    params = {}
    if implYy:
        params["implYy"] = implYy
    if implSeq:
        params["implSeq"] = implSeq

    status_code, data = await make_qnet_request(QNET_TEST_INFO_API, "getEList", params)
    return Response(content=data, media_type="application/xml", status_code=status_code)


@router.get("/c-list")
async def get_c_list(implYy: Optional[str] = None, implSeq: Optional[str] = None):
    """기능사 시험 시행일정 조회"""
    params = {}
    if implYy:
        params["implYy"] = implYy
    if implSeq:
        params["implSeq"] = implSeq

    status_code, data = await make_qnet_request(QNET_TEST_INFO_API, "getCList", params)
    return Response(content=data, media_type="application/xml", status_code=status_code)


@router.get("/fee-list")
async def get_fee_list(jmCd: str = Query(..., description="종목코드 (필수)")):
    """종목별 응시수수료 조회"""
    params = {"jmCd": jmCd}
    status_code, data = await make_qnet_request(QNET_TEST_INFO_API, "getFeeList", params)
    return Response(content=data, media_type="application/xml", status_code=status_code)


@router.get("/jm-list")
async def get_jm_list(
    jmCd: str = Query(..., description="종목코드 (필수)"),
    implYy: Optional[str] = Query(None, description="시행년도 (기본: 현재년도)")
):
    """종목별 시행일정 목록 조회"""
    from datetime import datetime

    params = {"jmCd": jmCd}

    # implYy가 없으면 현재 년도 사용
    if implYy:
        params["implYy"] = implYy
    else:
        params["implYy"] = str(datetime.now().year)

    status_code, data = await make_qnet_request(QNET_TEST_INFO_API, "getJMList", params)
    return Response(content=data, media_type="application/xml", status_code=status_code)


# ============================================
# 국가기술자격 종목 목록 조회 API
# ============================================

@router.get("/qualification-list")
async def get_qualification_list(gno: Optional[str] = None):
    """국가기술자격 종목 목록 조회"""
    params = {}
    if gno:
        params["gno"] = gno

    status_code, data = await make_qnet_request(QNET_QUALIFICATION_API, "getList", params)
    return Response(content=data, media_type="application/xml", status_code=status_code)
