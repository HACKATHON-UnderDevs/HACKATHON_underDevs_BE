import json
from typing import List
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from asyncio import sleep
import uuid
import shutil
import os
from markitdown import MarkItDown
from dotenv import load_dotenv
from convert import clean_json_string
from functions import create_prompt
from pydantic import BaseModel
import functions

app = FastAPI()
md = MarkItDown()

load_dotenv()
gemini_api_key = os.environ.get("GEMINI_API_KEY", "empty")
if not gemini_api_key or gemini_api_key == "empty":
    raise ValueError("GOOGLE_API_KEY environment variable is not set or is empty.")

client = genai.Client(api_key=gemini_api_key)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Welcome to the FastAPI application!"}


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
            config=functions.config,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
