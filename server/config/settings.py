"""
Configuration settings for the Study Helper API
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Q-Net API Configuration
QNET_TEST_INFO_API = "http://openapi.q-net.or.kr/api/service/rest/InquiryTestInformationNTQSVC"
QNET_QUALIFICATION_API = "http://openapi.q-net.or.kr/api/service/rest/InquiryListNationalQualifcationSVC"
QNET_SERVICE_KEY = "892a0f45b2c4a8b1b0c247d38df541b7d5d3ea40e069481b41424cdd1a77bc54"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 3001

# PDF Upload Configuration
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
