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
from datetime import datetime
from dataclasses import dataclass

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
    question_count: int = 5


class CreateFlashcardsRequest(BaseModel):
    flashcard_set_id: str
    note_content: str
    card_count: int = 10


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


class CreateQuizRequest(BaseModel):
    title: str
    subject: str
    user_id: str
    note_id: str
    note_content: str
    question_count: int = 5


@dataclass
class QuizAnswer:
    """Represents a quiz answer option"""
    option_text: str
    is_correct: bool
    answer_order: int


@dataclass
class QuizQuestion:
    """Represents a quiz question with multiple answers"""
    question_text: str
    question_type: str
    question_order: int
    answers: List[QuizAnswer]


@dataclass
class Quiz:
    """Represents a complete quiz with metadata and questions"""
    quiz_id: str
    title: str
    subject: str
    user_id: str
    note_id: str
    questions: List[QuizQuestion]


def generate_quiz_id() -> str:
    """Generate a unique quiz ID"""
    timestamp = int(datetime.now().timestamp() * 1000)
    random_suffix = uuid.uuid4().hex[:8]
    return f"quiz_{timestamp}_{random_suffix}"


def create_quiz_from_content(title: str, subject: str, user_id: str, note_id: str,
                           note_content: str, question_count: int = 5) -> Quiz:
    """Create a complete quiz from note content"""
    # Generate unique quiz ID
    quiz_id = generate_quiz_id()
    
    # Use the existing quiz generation logic
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=functions.create_quizzes_on_notes_prompt(
            note_content, functions.quiz_response_format, question_count
        ),
    )
    
    quizzes_str = clean_json_string(response.text)
    quizzes = json.loads(quizzes_str)
    
    # Parse backend response
    questions = []
    for idx, q_data in enumerate(quizzes):
        answers = []
        for ans_idx, answer_data in enumerate(q_data.get("answers", [])):
            answers.append(QuizAnswer(
                option_text=answer_data["option_text"],
                is_correct=answer_data["is_correct"],
                answer_order=ans_idx + 1
            ))
        
        questions.append(QuizQuestion(
            question_text=q_data["question_text"],
            question_type=q_data.get("question_type", "multiple_choice"),
            question_order=idx + 1,
            answers=answers
        ))
    
    return Quiz(
        quiz_id=quiz_id,
        title=title,
        subject=subject,
        user_id=user_id,
        note_id=note_id,
        questions=questions
    )


@app.post("/quizzes")
async def generate_quizzes_on_notes(request: CreateQuizzesRequest):
    print(request.note_content, functions.quiz_response_format)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=functions.create_quizzes_on_notes_prompt(
            request.note_content, functions.quiz_response_format, request.question_count
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


@app.post("/flashcards")
async def generate_flashcards_on_notes(request: CreateFlashcardsRequest):
    try:
        print(f"Generating flashcards for set: {request.flashcard_set_id}")
        print(f"Note content: {request.note_content[:200]}...")  # Print first 200 chars

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=functions.create_flashcards_on_notes_prompt(request.note_content, request.card_count),
        )

        flashcards_str = clean_json_string(response.text)
        print(f"Generated flashcards response: {flashcards_str}")

        flashcards = json.loads(flashcards_str)

        # Update each flashcard with the provided flashcard_set_id
        for flashcard in flashcards:
            flashcard["flashcard_set_id"] = request.flashcard_set_id
            print(f"Flashcard: {flashcard}")
            print("---------------------------------------------------------------------")

        return {"flashcards": flashcards}

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw response: {flashcards_str}")
        raise HTTPException(status_code=500, detail="Failed to parse AI response")
    except Exception as e:
        print(f"Error generating flashcards: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating flashcards: {str(e)}")


@app.post("/quizzes/create")
async def create_quiz(request: CreateQuizRequest):
    """Create a quiz from note content and return it as JSON for frontend to handle"""
    try:
        print(f"Creating quiz: {request.title}")
        print(f"User ID: {request.user_id}")
        print(f"Subject: {request.subject}")
        print(f"Question count: {request.question_count}")
        
        # Create quiz from content
        quiz = create_quiz_from_content(
            title=request.title,
            subject=request.subject,
            user_id=request.user_id,
            note_id=request.note_id,
            note_content=request.note_content,
            question_count=request.question_count
        )
        
        # Return quiz data for frontend to handle database operations
        return {
            "success": True,
            "quiz": {
                "quiz_id": quiz.quiz_id,
                "title": quiz.title,
                "subject": quiz.subject,
                "user_id": quiz.user_id,
                "note_id": quiz.note_id,
                "question_count": len(quiz.questions),
                "questions": [
                    {
                        "question_text": q.question_text,
                        "question_type": q.question_type,
                        "question_order": q.question_order,
                        "answers": [
                            {
                                "option_text": a.option_text,
                                "is_correct": a.is_correct,
                                "answer_order": a.answer_order
                            } for a in q.answers
                        ]
                    } for q in quiz.questions
                ]
            }
        }
        
    except Exception as e:
        print(f"Error creating quiz: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating quiz: {str(e)}")





if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
