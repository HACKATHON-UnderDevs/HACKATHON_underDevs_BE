import datetime
from google import genai
from google.genai import models, types
import json

# Define the function declaration for the model
meeting_function = {
    "name": "book_a_meeting",
    "description": "Books a meeting with the specified details.",
    "parameters": {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "The date of the meeting",
            },
            "time": {
                "type": "string",
                "description": "The time of the meeting, e.g. 14:00",
            },
            "topic": {
                "type": "string",
                "description": "The topic of the meeting, e.g. Project Update",
            },
        },
        "required": ["date", "time", "topic"],
    },
}

generate_quizzes_on_document_function = {
    "name": "generate_quizzes_on_document",
    "description": "Generates quizzes based on the provided document.",
    "parameters": {
        "type": "object",
        "properties": {
            "notes": {
                "type": "string",
                "description": "The date of the meeting",
            },
        },
        "required": ["notes"],
    },
}

quiz_response_format = """
[
  {
    "quiz_id": "none",
    "question_text": "What is the primary focus of biology?",
    "question_type": "multiple_choice",
    "answers": [
      {"option_text": "The study of inanimate objects", "is_correct": false},
      {"option_text": "The scientific study of life and living organisms", "is_correct": true},
      {"option_text": "The study of celestial bodies", "is_correct": false}
    ]
  },
  {
    "quiz_id": "none",
    "question_text": "Which of the following is a fundamental theme of biology?",
    "question_type": "multiple_choice",
    "answers": [
      {"option_text": "The cell as the basic unit of life", "is_correct": true},
      {"option_text": "The study of weather patterns", "is_correct": false},
      {"option_text": "The analysis of financial markets", "is_correct": false}
    ]
  }
]
"""


def create_prompt(user_input: str) -> str:
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"Today is {current_date}. {user_input}"


def create_document_summarize_prompt(document: str) -> str:
    return f"""Summarize the following document with BlockNote schema. Here is the BlockNote schema format:

    [
      {{
        type: "paragraph",
        content: "Welcome to this demo!",
      }},
      {{
        type: "paragraph",
      }},
      {{
        type: "paragraph",
        content: [
          {{
            type: "text",
            text: "Blocks:",
            styles: {{ bold: true }},
          }},
        ],
      }},
      {{
        type: "paragraph",
        content: "Paragraph",
      }},
      {{
        type: "heading",
        content: "Heading",
      }},
      {{
        id: "toggle-heading",
        type: "heading",
        props: {{ isToggleable: true }},
        content: "Toggle Heading",
      }},
      {{
        type: "quote",
        content: "Quote",
      }},
      {{
        type: "bulletListItem",
        content: "Bullet List Item",
      }},
      {{
        type: "numberedListItem",
        content: "Numbered List Item",
      }},
      {{
        type: "checkListItem",
        content: "Check List Item",
      }},
      {{
        id: "toggle-list-item",
        type: "toggleListItem",
        content: "Toggle List Item",
      }},
      {{
        type: "codeBlock",
        props: {{ language: "javascript" }},
        content: "console.log('Hello, world!');",
      }},
      {{
        type: "table",
        content: {{
          type: "tableContent",
          rows: [
            {{
              cells: ["Table Cell", "Table Cell", "Table Cell"],
            }},
            {{
              cells: ["Table Cell", "Table Cell", "Table Cell"],
            }},
            {{
              cells: ["Table Cell", "Table Cell", "Table Cell"],
            }},
          ],
        }},
      }},
      {{
        type: "file",
      }},
      {{
        type: "image",
        props: {{
          url: "https://interactive-examples.mdn.mozilla.net/media/cc0-images/grapefruit-slice-332-332.jpg",
          caption: "From https://interactive-examples.mdn.mozilla.net/media/cc0-images/grapefruit-slice-332-332.jpg",
        }},
      }},
      {{
        type: "video",
        props: {{
          url: "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.webm",
          caption: "From https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.webm",
        }},
      }},
      {{
        type: "audio",
        props: {{
          url: "https://interactive-examples.mdn.mozilla.net/media/cc0-audio/t-rex-roar.mp3",
          caption: "From https://interactive-examples.mdn.mozilla.net/media/cc0-audio/t-rex-roar.mp3",
        }},
      }},
      {{
        type: "paragraph",
      }},
      {{
        type: "paragraph",
        content: [
          {{
            type: "text",
            text: "Inline Content:",
            styles: {{ bold: true }},
          }},
        ],
      }},
      {{
        type: "paragraph",
        content: [
          {{
            type: "text",
            text: "Styled Text",
            styles: {{
              bold: true,
              italic: true,
              textColor: "red",
              backgroundColor: "blue",
            }},
          }},
          {{
            type: "text",
            text: " ",
            styles: {{}},
          }},
          {{
            type: "link",
            content: "Link",
            href: "https://www.blocknotejs.org",
          }},
        ],
      }},
      {{
        type: "paragraph",
      }},
    ]

    {document}"""


def create_quizzes_on_notes_prompt(notes: str, response_format: str) -> str:
    prompt = f"""Generate quizzes based on this content as JSON format. Here is the example response:
    {quiz_response_format}
    Here is the content: {notes}"""
    return prompt


# Configure the client and tools
client = genai.Client(api_key="AIzaSyDoGgguMGOlyTmJUxBMnEh7Frfb9GJWJCU")
tools = types.Tool(function_declarations=[generate_quizzes_on_document_function])
config = types.GenerateContentConfig(
    tools=[tools],
)


response = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents=create_prompt("What is the weather like in San Francisco?"),
    config=config,
)


def book_a_meeting(date: str, time: str, topic: str) -> str:
    result = {
        "status": "success",
        "meeting": {"date": date, "time": time, "topic": topic},
    }
    res = json.dumps(result)
    print(res)
    return res
