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
    {response_format}
    Here is the content: {notes}"""
    return prompt


def create_flashcards_on_notes_prompt(notes: str) -> str:
    prompt = f"""Generate flashcards based on this content as JSON format. Create question-answer pairs that help with learning and memorization.
    
    Return the response in this exact JSON format:
    [
      {{
        "flashcard_set_id": "none",
        "question": "What is the main concept?",
        "answer": "The detailed explanation of the main concept"
      }},
      {{
        "flashcard_set_id": "none", 
        "question": "Define key term X",
        "answer": "Key term X means..."
      }}
    ]
    
    Guidelines:
    - Create 5-10 flashcards
    - Questions should be clear and specific
    - Answers should be concise but complete
    - Focus on key concepts, definitions, and important facts
    - Avoid yes/no questions
    
    Here is the content: {notes}"""
    return prompt


def create_study_schedule_prompt(
    note_title: str, note_content: str, start_date: str, end_date: str
) -> str:
    prompt = """
    You are a highly capable assistant tasked with generating a study schedule in JSON format based on a provided `CreateStudySchedulesRequest` object. The input includes:

    - `note_title`: A string representing the title of the study schedule.
    - `note_content`: A JSON array of objects, where each object represents a content block (e.g., heading or paragraph) with properties like `id`, `type`, `props`, `content`, and `children`. Headings have `type: "heading"` and a `props.level` indicating their level (e.g., 2 or 3).
    - `startDate`: The start date of the study schedule (in YYYY-MM-DD format).
    - `endDate`: The end date of the study schedule (in YYYY-MM-DD format).

    Your task is to create a study schedule in the following JSON format:

    [
    {
        "title": "<note_title>",
        "part": "<heading text>",
        "dueDate": "<YYYY-MM-DD>",
        "priority": "<high|medium|low>",
        "count": <integer>,
        "estimatedTime": <integer>
    },
    ...
    ]

    **Requirements:**

    1. **Extract Headings**: Identify all objects in `note_content` with `type: "heading"`. Use the `text` field from the `content` array of these objects as the `part` value in the output.
    2. **Title**: Set the `title` field of each schedule entry to the provided `note_title`.
    3. **Due Dates**: Distribute due dates evenly between `startDate` and `endDate` (inclusive) for each heading. Ensure dates are in YYYY-MM-DD format and assigned sequentially.
    4. **Priority**: Assign priorities (`high`, `medium`, `low`) based on the importance of the content under each heading, not the heading level. Evaluate importance by analyzing the content of paragraphs following each heading. For example:
    - Content introducing core concepts or foundational knowledge (e.g., definitions, key principles) is `high` priority.
    - Content discussing benefits, advantages, or secondary details is `medium` priority.
    - Content covering specific features, tools, or less critical details is `low` priority.
    5. **Count and Estimated Time**:
    - `count`: Estimate the number of study tasks or items based on the content length or complexity under each heading (e.g., number of paragraphs or sentences). Use reasonable heuristics (e.g., 5–20 tasks per heading based on content size).
    - `estimatedTime`: Estimate the time (in minutes) required for studying the content under each heading. Base this on content length or complexity (e.g., 10–30 minutes per heading).
    6. **Validation**:
    - Ensure `startDate` is not later than `endDate`.
    - If no headings are found in `note_content`, return an empty array.
    - Handle malformed input gracefully (e.g., missing fields or invalid dates).
    7. **Output**: Return a JSON array of schedule entries, sorted by `dueDate`.

    **Example Input:**

    {
    "note_title": "C# Learning",
    "note_content": [
        {
        "id": "2f196ccd-ebcd-4a53-95ce-aeb245742494",
        "type": "heading",
        "props": { "textColor": "default", "backgroundColor": "default", "textAlignment": "left", "level": 2, "isToggleable": false },
        "content": [ { "type": "text", "text": "REST APIs with .NET and C#", "styles": { "bold": true } } ],
        "children": []
        },
        {
        "id": "2b7a6170-db1a-4901-895a-74df2b51e3c6",
        "type": "paragraph",
        "props": { "textColor": "default", "backgroundColor": "default", "textAlignment": "left" },
        "content": [ { "type": "text", "text": "REST (Representational State of Resource) APIs, or Application Programming Interfaces, are used for building services that can be consumed by a wide range of clients, including web browsers and mobile devices. They rely on a stateless, client-server, cacheable communications protocol, and in the case of .NET and C#, are typically built using the ASP.NET framework.", "styles": {} } ],
        "children": []
        },
        {
        "id": "cba82c6c-8e37-435e-95d6-1989f0bbfc81",
        "type": "heading",
        "props": { "textColor": "default", "backgroundColor": "default", "textAlignment": "left", "level": 3, "isToggleable": false },
        "content": [ { "type": "text", "text": "Benefits of Using .NET and C# for REST APIs", "styles": {} } ],
        "children": []
        },
        {
        "id": "b18f92cd-94ea-4832-b03b-3f6b560a41b3",
        "type": "paragraph",
        "props": { "textColor": "default", "backgroundColor": "default", "textAlignment": "left" },
        "content": [ { "type": "text", "text": "With ASP.NET you use the same framework and patterns to build both web pages and services, side-by-side in the same project.", "styles": {} }, { "type": "link", "href": "http://ASP.NET", "content": [ { "type": "text", "text": "ASP.NET", "styles": {} } ] }, { "type": "text", "text": " you use the same framework and patterns to build both web pages and services, side-by-side in the same project.", "styles": {} } ],
        "children": []
        },
        {
        "id": "1d2b4bcc-9271-4992-a85d-52589afb7d81",
        "type": "heading",
        "props": { "textColor": "default", "backgroundColor": "default", "textAlignment": "left", "level": 3, "isToggleable": false },
        "content": [ { "type": "text", "text": "Key Features of REST APIs with .NET and C#", "styles": {} } ],
        "children": []
        },
        {
        "id": "466c86ab-8cc7-4f19-982f-d91e2e9727ee",
        "type": "paragraph",
        "props": { "textColor": "default", "backgroundColor": "default", "textAlignment": "left" },
        "content": [ { "type": "text", "text": "Some key features of REST APIs built using .NET and C# include support for HTTP methods such as GET, POST, PUT, and DELETE, as well as support for JSON and XML data formats. Additionally, .NET and C# provide a range of tools and libraries for building and consuming REST APIs, including ASP.NET Web API and HttpClient.", "styles": {} } ],
        "children": []
        },
        ...
    ],
    "startDate": "2025-10-25",
    "endDate": "2025-10-30"
    }

    **Example Output:**

    [
    {
        "title": "C# Learning",
        "part": "REST APIs with .NET and C#",
        "dueDate": "2025-10-25",
        "priority": "high",
        "count": 15,
        "estimatedTime": 20
    },
    {
        "title": "C# Learning",
        "part": "Benefits of Using .NET and C# for REST APIs",
        "dueDate": "2025-10-27",
        "priority": "medium",
        "count": 10,
        "estimatedTime": 15
    },
    {
        "title": "C# Learning",
        "part": "Key Features of REST APIs with .NET and C#",
        "dueDate": "2025-10-29",
        "priority": "low",
        "count": 12,
        "estimatedTime": 18
    }
    ]

    **Instructions:**

    - Process the provided input and generate a study schedule following the above requirements.
    - Use reasonable heuristics for `count` and `estimatedTime` based on the content under each heading.
    - Ensure the output is a valid JSON array, properly formatted.
    - If any issues arise (e.g., invalid dates or missing headings), return an empty array or handle gracefully with appropriate defaults.

    **Input to Process:**

    {
    "note_title": "C# Learning",
    "note_content": <provided note_content>,
    "startDate": "2025-10-25",
    "endDate": "2025-10-30"
    }

    Generate the study schedule in JSON format.
    """
    return prompt


def book_a_meeting(date: str, time: str, topic: str) -> str:
    result = {
        "status": "success",
        "meeting": {"date": date, "time": time, "topic": topic},
    }
    res = json.dumps(result)
    print(res)
    return res
