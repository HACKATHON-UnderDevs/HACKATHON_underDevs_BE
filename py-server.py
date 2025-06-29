from datetime import date
from functools import partial
import json
import sys
from typing import List
from fastapi import FastAPI, HTTPException, File, UploadFile, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    Stream,
    get_cloudflare_turn_credentials,
    get_cloudflare_turn_credentials_async,
)
from google import genai
from google.genai import types
from asyncio import sleep
import uuid
import shutil
import os
import gradio
from markitdown import MarkItDown
from dotenv import load_dotenv
from convert import clean_json_string
from functions import create_prompt
from pydantic import BaseModel
import functions
from groq import Groq
from elevenlabs import ElevenLabs
import numpy as np

import voice

app = FastAPI()
md = MarkItDown()

load_dotenv()
gemini_api_key = os.environ.get("GEMINI_API_KEY", "empty")
if not gemini_api_key or gemini_api_key == "empty":
    raise ValueError("GEMINI_API_KEY environment variable is not set or is empty.")


client = genai.Client(api_key=gemini_api_key)
groq_client = voice.groq_client
tts_client = voice.tts_client
hf_token = os.environ.get("HUGGING_FACE_API_KEY", "empty")
if not hf_token or hf_token == "empty":
    raise ValueError("HUGGINGFACE environment variable is not set or is empty.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Cloudflare TURN credentials function
async def get_turn_credentials():
    """Get Cloudflare TURN credentials with HF token"""
    if hf_token:
        try:
            return await get_cloudflare_turn_credentials_async(hf_token=hf_token)
        except Exception as e:
            # Fallback to basic STUN servers
            return {
                "iceServers": [
                    {"urls": "stun:stun.l.google.com:19302"},
                    {"urls": "stun:global.stun.twilio.com:3478"},
                ]
            }
    else:
        # Fallback configuration
        return {
            "iceServers": [
                {"urls": "stun:stun.l.google.com:19302"},
                {"urls": "stun:global.stun.twilio.com:3478"},
                {"urls": "stun:stun.services.mozilla.com:3478"},
            ]
        }


@app.get("/")
def root():
    return {"message": "Welcome to the FastAPI application!"}


class StartSessionRequest(BaseModel):
    note_content: str
    title: str = "Learning with Vibe Learning"


async def get_credentials():
    return await get_cloudflare_turn_credentials_async(hf_token=hf_token)


@app.post("/start-voice-session")
async def start_voice_session(request: StartSessionRequest):
    try:
        print("hf_token:", hf_token)
        # Create handler with context
        handler_with_context = partial(
            voice.voice_teacher_handler, note_content=request.note_content
        )

        stream = Stream(
            handler=ReplyOnPause(handler_with_context, input_sample_rate=16000),
            modality="audio",
            mode="send-receive",
            rtc_configuration=get_turn_credentials,  # Async function for client
            ui_args={
                "title": request.title,
                "chatbot_initial": [
                    {
                        "role": "assistant",
                        "content": "Hello! I'm ready to help you review your notes. What would you like to go over first?",
                    }
                ],
            },
        )

        # Launch with enhanced configuration
        share_url = stream.ui.launch(
            share=True,
            strict_cors=False,
            server_name="0.0.0.0",
            server_port=None,  # Let Gradio choose
            ssl_verify=False,
        )

        return {
            "session_url": share_url,
            "turn_provider": "cloudflare" if hf_token else "fallback",
            "rtc_config": "cloudflare_enhanced",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create session: {str(e)}"
        )


@app.post("/documents")
async def generate_note_from_documents(file: UploadFile = File(...)):
    unique_id = uuid.uuid4()
    temp_dir = f"./temp/{unique_id}"
    try:
        # Create temp directory
        os.makedirs(temp_dir, exist_ok=True)

        # Save uploaded file
        file_path = f"{temp_dir}/{file.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Convert file to Markdown (assuming MarkItDown handles the file path)
        result = md.convert(file_path)
        content = result.text_content

        # Summarize with Gemini API
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=functions.create_document_summarize_prompt(content),
        )
        summary = response.text  # Adjust based on actual response structure

        print(f"Generated summary: {summary}")
        print(f"Generated content: {content}")

        # Clean up temp directory
        shutil.rmtree(temp_dir)

        return {"summary": clean_json_string(summary)}

    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


class CreateQuizzesRequest(BaseModel):
    quiz_id: str
    note_content: str


class CreateFlashcardsRequest(BaseModel):
    flashcard_set_id: str
    note_content: str


class FlashcardResponse(BaseModel):
    flashcard_set_id: str
    question: str
    answer: str


class AnswerResponse:
    options_text: str
    is_correct: str


class QuestionResponse:
    quiz_id: str
    question_text: str
    question_type: str
    answers: List[AnswerResponse]


@app.post("/quizzes")
async def generate_quizzes_on_notes(request: CreateQuizzesRequest):
    print(request.note_content, functions.quiz_response_format)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=functions.create_quizzes_on_notes_prompt(
            request.note_content, functions.quiz_response_format
        ),
    )

    quizzes_str = clean_json_string(response.text)
    print(quizzes_str)

    quizzes = json.loads(quizzes_str)

    for quiz in quizzes:
        quiz["quiz_id"] = request.quiz_id
        print(f"{quiz}\n")
        print("---------------------------------------------------------------------\n")

    return {"quizzes": quizzes}


class CreateStudySchedulesRequest(BaseModel):
    note_content: str
    note_title: str
    startDate: str
    endDate: str


@app.post("/study-sets")
async def generate_study_schedules_on_notes(request: CreateStudySchedulesRequest):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=functions.create_study_schedules_on_notes_prompt(
            request.note_content,
            request.note_title,
            request.startDate,
            request.endDate,
        ),
    )

    schedules_str = clean_json_string(response.text)
    print(schedules_str)

    # schedules = json.loads(schedules_str)

    return {"study_sets": schedules_str}


@app.post("/flashcards")
async def generate_flashcards_on_notes(request: CreateFlashcardsRequest):
    try:
        print(f"Generating flashcards for set: {request.flashcard_set_id}")
        print(f"Note content: {request.note_content[:200]}...")  # Print first 200 chars

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=functions.create_flashcards_on_notes_prompt(request.note_content),
        )

        flashcards_str = clean_json_string(response.text)
        print(f"Generated flashcards response: {flashcards_str}")

        flashcards = json.loads(flashcards_str)

        # Update each flashcard with the provided flashcard_set_id
        for flashcard in flashcards:
            flashcard["flashcard_set_id"] = request.flashcard_set_id
            print(f"Flashcard: {flashcard}")
            print(
                "---------------------------------------------------------------------"
            )

        return {"flashcards": flashcards}

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw response: {flashcards_str}")
        raise HTTPException(status_code=500, detail="Failed to parse AI response")
    except Exception as e:
        print(f"Error generating flashcards: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating flashcards: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
