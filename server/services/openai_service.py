"""
OpenAI Service
Handles all OpenAI API interactions
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
from openai import OpenAI
from fastapi import HTTPException

from config.settings import OPENAI_API_KEY


# Initialize OpenAI client
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None
    print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")


def format_date(date_str: str) -> str:
    """Format date string from yyyymmdd to yyyy년 mm월 dd일"""
    if not date_str or len(date_str) != 8:
        return date_str

    try:
        year = date_str[0:4]
        month = date_str[4:6]
        day = date_str[6:8]
        return f"{year}년 {month}월 {day}일"
    except:
        return date_str


async def generate_study_plan(subject: str, exam_schedule: Dict[str, Any], start_date: str) -> Dict[str, Any]:
    """
    Generate AI-powered study plan based on exam information and start date

    Args:
        subject: Exam subject name
        exam_schedule: Dictionary containing exam schedule information
        start_date: Study start date (YYYY-MM-DD format)

    Returns:
        Dictionary containing study plan and metadata
    """
    if not openai_client:
        raise HTTPException(
            status_code=503,
            detail="OpenAI service is not available. Please set OPENAI_API_KEY in .env file"
        )

    if not subject:
        raise HTTPException(status_code=400, detail="Subject is required")

    # Format exam schedule information
    schedule_info = ""
    doc_exam_date = ""
    prac_exam_date = ""

    if exam_schedule:
        if exam_schedule.get("docRegStartDt"):
            schedule_info += f"필기시험 원서접수: {format_date(exam_schedule.get('docRegStartDt'))} ~ {format_date(exam_schedule.get('docRegEndDt', ''))}\n"
        if exam_schedule.get("docExamDt"):
            doc_exam_date = exam_schedule.get('docExamDt')
            schedule_info += f"필기시험 일자: {format_date(doc_exam_date)}\n"
        if exam_schedule.get("docPassDt"):
            schedule_info += f"필기시험 합격자 발표: {format_date(exam_schedule.get('docPassDt'))}\n"
        if exam_schedule.get("pracRegStartDt"):
            schedule_info += f"실기시험 원서접수: {format_date(exam_schedule.get('pracRegStartDt'))} ~ {format_date(exam_schedule.get('pracRegEndDt', ''))}\n"
        if exam_schedule.get("pracExamStartDt"):
            prac_exam_date = exam_schedule.get('pracExamStartDt')
            schedule_info += f"실기시험 기간: {format_date(prac_exam_date)} ~ {format_date(exam_schedule.get('pracExamEndDt', ''))}\n"
        if exam_schedule.get("pracPassDt"):
            schedule_info += f"최종 합격자 발표: {format_date(exam_schedule.get('pracPassDt'))}\n"

    # Calculate study period and weekly schedule
    study_period_info = ""
    written_exam_schedule = ""
    practical_exam_schedule = ""

    if start_date:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            start_date_kr = start_dt.strftime("%Y년 %m월 %d일")

            # Calculate days until written exam
            if doc_exam_date:
                doc_exam_dt = datetime.strptime(doc_exam_date, "%Y%m%d")
                days_to_doc = (doc_exam_dt - start_dt).days
                weeks_to_doc = days_to_doc // 7

                study_period_info += f"\n공부 시작일: {start_date_kr}\n"
                study_period_info += f"필기시험일: {format_date(doc_exam_date)}\n"
                study_period_info += f"필기시험까지 남은 기간: {days_to_doc}일 (약 {weeks_to_doc}주)\n"

                # Generate weekly schedule for written exam
                written_exam_schedule = "\n### 필기시험 대비 주차별 일정\n\n"
                current_date = start_dt
                for week in range(1, weeks_to_doc + 1):
                    week_start = current_date
                    week_end = current_date + timedelta(days=6)

                    # Don't go past the exam date
                    if week_end >= doc_exam_dt:
                        week_end = doc_exam_dt - timedelta(days=1)

                    written_exam_schedule += f"- **{week}주차**: {week_start.strftime('%m-%d')} ~ {week_end.strftime('%m-%d')}\n"
                    current_date += timedelta(days=7)

                # Add D-day countdown
                written_exam_schedule += f"\n- **시험 전 주**: {(doc_exam_dt - timedelta(days=7)).strftime('%m-%d')} ~ {(doc_exam_dt - timedelta(days=1)).strftime('%m-%d')} (D-7 ~ D-1)\n"
                written_exam_schedule += f"- **시험 당일**: {doc_exam_dt.strftime('%m-%d')} (D-Day)\n"

            # Calculate days until practical exam
            if prac_exam_date:
                prac_exam_dt = datetime.strptime(prac_exam_date, "%Y%m%d")

                # Start from day after written exam (or after result announcement)
                prac_start_dt = doc_exam_dt + timedelta(days=1) if doc_exam_date else start_dt

                if exam_schedule.get("docPassDt"):
                    doc_pass_dt = datetime.strptime(exam_schedule.get("docPassDt"), "%Y%m%d")
                    prac_start_dt = doc_pass_dt + timedelta(days=1)

                days_to_prac = (prac_exam_dt - prac_start_dt).days
                weeks_to_prac = days_to_prac // 7

                study_period_info += f"\n실기시험 준비 시작일: {prac_start_dt.strftime('%Y년 %m월 %d일')}\n"
                study_period_info += f"실기시험일: {format_date(prac_exam_date)}\n"
                study_period_info += f"실기시험까지 남은 기간: {days_to_prac}일 (약 {weeks_to_prac}주)\n"

                # Generate weekly schedule for practical exam
                practical_exam_schedule = "\n### 실기시험 대비 주차별 일정\n\n"
                current_date = prac_start_dt
                for week in range(1, weeks_to_prac + 1):
                    week_start = current_date
                    week_end = current_date + timedelta(days=6)

                    # Don't go past the exam date
                    if week_end >= prac_exam_dt:
                        week_end = prac_exam_dt - timedelta(days=1)

                    practical_exam_schedule += f"- **{week}주차**: {week_start.strftime('%m-%d')} ~ {week_end.strftime('%m-%d')}\n"
                    current_date += timedelta(days=7)

                # Add D-day countdown
                practical_exam_schedule += f"\n- **시험 전 주**: {(prac_exam_dt - timedelta(days=7)).strftime('%m-%d')} ~ {(prac_exam_dt - timedelta(days=1)).strftime('%m-%d')} (D-7 ~ D-1)\n"
                practical_exam_schedule += f"- **시험 당일**: {prac_exam_dt.strftime('%m-%d')} (D-Day)\n"

        except Exception as e:
            print(f"Date calculation error: {e}")

    # Create prompt for OpenAI
    prompt = f"""당신은 국가기술자격 시험 전문 학습 컨설턴트입니다.

시험 종목: {subject}

시험 일정:
{schedule_info if schedule_info else "일정 정보가 제공되지 않았습니다."}
{study_period_info}
{written_exam_schedule}
{practical_exam_schedule}

위 정보를 바탕으로 수험생을 위한 맞춤형 학습 계획을 작성해주세요.
**중요: 위에 제공된 주차별 일정의 정확한 날짜를 반드시 사용하세요.**

다음 내용을 반드시 포함해주세요:

## 1. 시험 개요 및 난이도 분석
- 이 자격증의 특징과 산업 내 위치
- 평균 합격률과 난이도 수준
- 권장 준비 기간

## 2. 필기시험 준비 전략
- 주요 출제 과목 (예: 과목명 1, 과목명 2, 과목명 3...)
- 과목별 출제 비중과 난이도
- 과목별 학습 방법 및 핵심 포인트
- 추천 교재 및 온라인 강의
- 문제풀이 전략

## 3. 필기시험 대비 주차별 상세 학습 계획

**위에 제공된 필기시험 주차별 일정에 맞춰 다음과 같이 표 형식으로 작성해주세요 (날짜는 MM-DD 형식으로):**

| 주차 | 기간 | 학습 주제 | 주요 학습 내용 | 일일 학습량 | 체크포인트 |
|------|------|-----------|---------------|-------------|-----------|
| 1주차 | MM-DD ~ MM-DD | [과목/챕터명] | - 월: 내용<br>- 화: 내용<br>- 수: 내용<br>- 목: 내용<br>- 금: 내용<br>- 주말: 복습 | 하루 2-3시간 | ✅ 확인사항 |
| 2주차 | MM-DD ~ MM-DD | [과목/챕터명] | ... | ... | ... |

**시험 전 주 집중 관리 (D-7 ~ D-Day):**

| 날짜 | D-Day | 집중 학습 내용 | 시간 배분 | 주의사항 |
|------|-------|---------------|-----------|----------|
| MM-DD | D-7 | 핵심 개념 총정리 | 3-4시간 | ... |
| MM-DD | D-5 | 취약 과목 집중 | 3-4시간 | ... |
| MM-DD | D-3 | 모의고사 풀이 | 실전 시간 | ... |
| MM-DD | D-1 | 최종 암기 사항 정리 | 2시간 | 일찍 취침 |
| MM-DD | D-Day | 시험 당일 | - | 준비물 확인 |

## 4. 실기시험 준비 전략
- 실기 과제 유형 및 출제 경향
- 실습 환경 구축 방법
- 필수 암기 사항 및 공식
- 실전 연습 방법
- 시간 관리 팁

## 5. 실기시험 대비 주차별 상세 학습 계획

**위에 제공된 실기시험 주차별 일정에 맞춰 다음과 같이 표 형식으로 작성해주세요 (날짜는 MM-DD 형식으로):**

| 주차 | 기간 | 실습 주제 | 주요 실습 내용 | 일일 실습량 | 체크포인트 |
|------|------|-----------|---------------|-------------|-----------|
| 1주차 | MM-DD ~ MM-DD | [과제 유형] | - 월: 실습 내용<br>- 화: 실습 내용<br>- 수: 실습 내용<br>- 목: 실습 내용<br>- 금: 실습 내용<br>- 주말: 종합 실습 | 하루 3-4시간 | ✅ 완성도 확인 |
| 2주차 | MM-DD ~ MM-DD | [과제 유형] | ... | ... | ... |

**시험 전 주 집중 관리 (D-7 ~ D-Day):**

| 날짜 | D-Day | 집중 실습 내용 | 시간 배분 | 주의사항 |
|------|-------|---------------|-----------|----------|
| MM-DD | D-7 | 핵심 과제 반복 실습 | 4-5시간 | 시간 재며 연습 |
| MM-DD | D-5 | 취약 과제 집중 | 4-5시간 | ... |
| MM-DD | D-3 | 모의 실기 시험 | 실전 시간 | ... |
| MM-DD | D-1 | 최종 점검 및 정리 | 2시간 | 일찍 취침 |
| MM-DD | D-Day | 시험 당일 | - | 실습 도구 확인 |

## 6. 최종 마무리 전략
- 시험 전날 준비사항
- 시험장 필수 준비물 체크리스트
- 시험 당일 시간 배분 전략
- 멘탈 관리 및 컨디션 유지 방법

**중요 작성 지침:**
- ✅ **반드시 위에 제공된 주차별 일정의 정확한 날짜를 사용하세요 (MM-DD 형식)**
- ✅ 필기시험과 실기시험 계획을 명확히 구분하여 작성하세요
- ✅ 모든 주차별 계획은 **표(table) 형식**으로 작성하세요
- ✅ 각 주차별로 매일(월~일)의 구체적인 학습/실습 내용과 분량을 명시하세요
- ✅ 학습 기간을 고려하여 현실적이고 실천 가능한 계획을 제시하세요
- ✅ 마크다운 형식으로 보기 좋게 작성하세요
- ✅ D-7, D-5, D-3, D-1 계획은 실제 날짜를 MM-DD 형식으로 작성하세요"""

    try:
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 국가기술자격 시험 전문 학습 컨설턴트입니다. 수험생들이 효율적으로 시험을 준비할 수 있도록 구체적이고 실용적인 조언을 제공합니다. 주어진 학습 기간에 맞춰 현실적이고 실천 가능한 일정을 제시합니다. 주차별 계획은 반드시 매일의 구체적인 학습 내용을 포함하여 상세하게 작성합니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )

        study_plan = response.choices[0].message.content

        return {
            "success": True,
            "subject": subject,
            "study_plan": study_plan,
            "exam_schedule": exam_schedule,
            "start_date": start_date
        }

    except Exception as e:
        print(f"OpenAI API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate study plan: {str(e)}")


async def chat_with_ai(messages: List[Dict[str, str]]) -> str:
    """
    Chat with AI assistant

    Args:
        messages: List of message dictionaries with 'role' and 'content'

    Returns:
        AI response message
    """
    if not openai_client:
        raise HTTPException(
            status_code=503,
            detail="OpenAI service is not available. Please set OPENAI_API_KEY in .env file"
        )

    if not messages:
        raise HTTPException(status_code=400, detail="Messages are required")

    try:
        # System message
        system_message = {
            "role": "system",
            "content": """당신은 친절하고 전문적인 학습 도우미 AI입니다.

주요 역할:
- 국가기술자격 시험 준비에 대한 조언 제공
- 학습 계획 수립 도움
- 시험 준비 방법 안내
- 학습 동기 부여 및 격려
- 학습 관련 질문에 대한 친절한 답변

답변 스타일:
- 친근하고 이해하기 쉬운 언어 사용
- 구체적이고 실용적인 조언 제공
- 긍정적이고 격려하는 태도 유지
- 필요시 단계별로 설명
- 이모지를 적절히 활용하여 친근감 표현

한국어로 답변해주세요."""
        }

        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[system_message] + messages,
            temperature=0.8,
            max_tokens=1000
        )

        ai_message = response.choices[0].message.content

        return ai_message

    except Exception as e:
        print(f"OpenAI Chat API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")


async def chat_with_ai_and_file(messages: List[Dict[str, str]], file_content: str, filename: str) -> str:
    """
    Chat with AI assistant including file content

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        file_content: Content of the uploaded file
        filename: Name of the uploaded file

    Returns:
        AI response message
    """
    if not openai_client:
        raise HTTPException(
            status_code=503,
            detail="OpenAI service is not available. Please set OPENAI_API_KEY in .env file"
        )

    try:
        # System message
        system_message = {
            "role": "system",
            "content": """당신은 친절하고 전문적인 학습 도우미 AI입니다.

주요 역할:
- 국가기술자격 시험 준비에 대한 조언 제공
- 학습 계획 수립 도움
- 시험 준비 방법 안내
- 학습 동기 부여 및 격려
- 학습 관련 질문에 대한 친절한 답변
- 업로드된 파일 내용 분석 및 피드백 제공

답변 스타일:
- 친근하고 이해하기 쉬운 언어 사용
- 구체적이고 실용적인 조언 제공
- 긍정적이고 격려하는 태도 유지
- 필요시 단계별로 설명
- 이모지를 적절히 활용하여 친근감 표현

한국어로 답변해주세요."""
        }

        # Add file content to the last user message
        file_context = f"\n\n[업로드된 파일: {filename}]\n```\n{file_content}\n```"

        # Combine messages with file content
        combined_messages = messages.copy()
        if combined_messages and combined_messages[-1]["role"] == "user":
            combined_messages[-1]["content"] += file_context

        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[system_message] + combined_messages,
            temperature=0.8,
            max_tokens=1500
        )

        ai_message = response.choices[0].message.content

        return ai_message

    except Exception as e:
        print(f"OpenAI Chat API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
