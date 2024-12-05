from typing import Optional
from fastapi import APIRouter, File, HTTPException, Body, UploadFile, status, Request
from loguru import logger
from datetime import datetime
import boto3
import json
import os
import uuid
import requests
import ssl
import base64
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from bs4 import BeautifulSoup
from oros.config import settings
from langchain_openai.chat_models import AzureChatOpenAI
from oros.models import QuizRecommendationTopics
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (PromptTemplate, 
                                    ChatPromptTemplate, HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)

from oros.prompts import *
from sse_starlette.sse import EventSourceResponse
from oros.db import *
from oros.tracker import track_event

BUCKET_NAME = os.getenv('WASABI_BUCKET')
router = APIRouter()

from dotenv import load_dotenv
load_dotenv()

ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
TOPIC = os.getenv("TOPIC")

async def chain_runnable(resume_text="", jd_text="", model_name="OpenAI", skills=[], fresher=False, trace_id=""):
    
    model = AzureChatOpenAI(
        temperature=0,
        model_kwargs={"seed": 42},
        streaming=True,
        azure_endpoint=settings.azure_openai_endpoint,
        openai_api_version=settings.azure_openai_api_version,
        openai_api_key=settings.azure_openai_key,
        deployment_name=settings.azure_openai_chatgpt_deployment,
        openai_api_type=settings.openai_api_type
    )
 
    combined_response = {}
    
    return combined_response
        
@router.post("/qa_with_topics")
async def evaluate_with_jd(request: Request,
                           quiz_duration: int = Body(None, description="Duration of the quiz in minutes"),
                           difficulty_level: str = Body(None, description="Level of difficulty - Easy, Medium, Hard"),
                           quiz_topics: list = Body(None, description="List of topics to be covered in the Quiz"),
                           quiz_type: str = Body(None, description="Quiz Type")):
    
    if not quiz_duration or not difficulty_level or not quiz_topics:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing required quiz parameters")
    document_id = str(uuid.uuid4())
    event_id = str(uuid.uuid4())
    body = {"id": document_id,
            "nr_questions": quiz_duration,
            "difficulty_level": difficulty_level,
            "topics": quiz_topics, 
            "quiz_type": quiz_type,
            "user_id": int(request.headers.get("Userid")),
            "workspace_id": int(request.headers.get("Workspaceid")),
            "event_id": event_id,
            "function": "qa_with_topics"
            }

    topic=TOPIC
    
    return {"message": "Quiz creation from document has been initiated", "doc_id": document_id, "event_id": event_id}

@router.post("/interviewer")
async def create_quiz_from_resume(request: Request,
                                  resume_file: str = Body(..., description="The document content as a string"),
                                  file_type: str = Body(None, description="File type as a string"),
                                  job_title: str = Body(None, description="Job Title as a string"),
                                  company_name: str = Body(None, description="company name as a string"),
                                  job_description: str = Body(None, description="job description as a string")):
    try:   
        
        
        return {"message": "Quiz creation from document has been initiated", "doc_id": document_id, "event_id": event_id}
    except Exception as e:
        logger.error(f"Error creating quiz from document: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process resume for quiz creation")

