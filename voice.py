import io
import os
import tempfile
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from fastrtc import AdditionalOutputs, Stream, ReplyOnPause
import numpy as np
import wave
import time

from numpy.typing import NDArray


load_dotenv()


def get_groq_client():
    """Initialize and return the Groq client"""
    from groq import Groq

    # Ensure the GROQ_API_KEY environment variable is set
    groq_api_key = os.environ.get("GROQ_API_KEY", "empty")
    if not groq_api_key or groq_api_key == "empty":
        raise ValueError("GROQ_API_KEY environment variable is not set or is empty.")
    return Groq(api_key=groq_api_key)


def get_tts_client():
    elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY", "empty")
    if not elevenlabs_api_key or elevenlabs_api_key == "empty":
        raise ValueError(
            "ELEVENLABS_API_KEY environment variable is not set or is empty."
        )

    return ElevenLabs(api_key=elevenlabs_api_key)


groq_client = get_groq_client()
tts_client = get_tts_client()


def audio_to_wav_file(audio_data: NDArray, sample_rate: int) -> bytes:
    """Convert audio data to a temporary WAV file for Groq processing"""
    with io.BytesIO() as wav_file:
        with wave.open(wav_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        return wav_file.getvalue()


def transcribe_with_groq(audio: tuple[int, NDArray]) -> str:
    """Transcribe audio using Groq's Whisper API"""
    sample_rate, audio_data = audio

    # Convert audio data to a temporary WAV file
    wav_bytes = audio_to_wav_file(audio_data, sample_rate)

    transcription = groq_client.audio.translations.create(
        file=("input.wav", wav_bytes),
        model="whisper-large-v3",
    )
    return transcription.text


def voice_teacher_handler(
    audio: tuple[int, NDArray[np.int16 | np.float32]],
    note_content: str,
    chatbot: list[dict] | None = None,
):
    """
    The main handler for the voice conversation.
    It transcribes user audio, gets a response from the LLM (acting as a teacher),
    and streams the audio response back.
    """
    chatbot = chatbot or []

    # 1. System Prompt Injection (Teacher Persona)
    # This runs only on the first turn of the conversation.
    if not chatbot:
        system_prompt = f"""
        You are a helpful and patient teacher. Your goal is to help the user review and learn the following material.
        Engage them in a conversation, ask them questions about the material to test their knowledge, and clarify concepts they are unsure about.
        Start by greeting the user and asking them what part of the note they'd like to begin with.

        --- MATERIAL TO REVIEW ---
        {note_content}
        --- END OF MATERIAL ---
        """
        chatbot.append({"role": "system", "content": system_prompt})

    # 2. Transcribe User's Audio
    start_time = time.time()
    try:
        user_text = transcribe_with_groq(audio)
        if not user_text.strip():
            print("No speech detected.")
            # Yield nothing to indicate no response is needed
            return

        print(f"Transcription ({time.time() - start_time:.2f}s): '{user_text}'")
    except Exception as e:
        print(f"Error during transcription: {e}")
        return  # Stop processing if transcription fails

    # 3. Update Chat History and Yield to UI
    chatbot.append({"role": "user", "content": user_text})
    yield AdditionalOutputs(chatbot)  # fastrtc feature to update state

    # Prepare messages for the LLM API
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in chatbot]

    # 4. Generate LLM Response
    print("Getting LLM response from Groq...")
    llm_start_time = time.time()
    response_stream = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        stream=True,  # Use streaming for lower first-token latency
    )

    # Stream the response text as it comes in
    response_text = ""
    for chunk in response_stream:
        delta = chunk.choices[0].delta.content
        if delta:
            response_text += delta

    print(f"LLM Response ({time.time() - llm_start_time:.2f}s): '{response_text}'")

    # Update chat history with the full assistant response
    chatbot.append({"role": "assistant", "content": response_text})

    # 5. Convert LLM Text to Speech and Stream Audio
    print("Streaming audio response from ElevenLabs...")
    tts_start_time = time.time()

    # The `output_format` pcm_24000 gives us a raw stream of 16-bit samples at 24000 Hz
    # This is efficient as we can yield chunks directly.
    audio_stream = tts_client.text_to_speech.stream(
        text=response_text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",  # A good, clear voice
        model_id="eleven_multilingual_v2",
        output_format="pcm_24000",
    )

    for chunk in audio_stream:
        # The chunk is already in bytes, convert to numpy array for fastrtc
        audio_array = np.frombuffer(chunk, dtype=np.int16)
        yield (24000, audio_array)  # Yield sample rate and audio chunk

    print(f"TTS streaming finished in {time.time() - tts_start_time:.2f}s")


# def generate_response(
#     audio: tuple[int, NDArray[np.int16 | np.float32]], chatbot: list[dict] | None = None
# ):
#     chatbot = chatbot or []
#     messages = [{"role": msg["role"], "content": msg["content"]} for msg in chatbot]
#     start = time.time()
#     text = transcribe_with_groq(audio)
#     print("transcription", time.time() - start)
#     print("prompt", text)
#
#     chatbot.append({"role": "user", "content": text})
#
#     yield AdditionalOutputs(chatbot)
#
#     messages.append({"role": "user", "content": text})
#
#     response_text = (
#         groq_client.chat.completions.create(
#             model="llama-3.1-8b-instant",
#             max_tokens=512,
#             messages=messages,
#         )
#         .choices[0]
#         .message.content
#     )
#
#     chatbot.append({"role": "assistant", "content": response_text})
#
#     yield response_text


def response(
    audio: tuple[int, NDArray[np.int16 | np.float32]], chatbot: list[dict] | None = None
):
    # Transcription and response generation
    gen = generate_response(audio, chatbot)

    # First yield is AdditionalOutputs with updated chatbot
    chatbot = next(gen)

    # Second yield is the response text
    response_text = next(gen)

    print(response_text)

    # Fall back to ElevenLabs for reliable TTS
    print("Using ElevenLabs for text-to-speech")
    for chunk in tts_client.text_to_speech.stream(
        text=response_text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="pcm_24000",
    ):
        audio_array = np.frombuffer(chunk, dtype=np.int16).reshape(1, -1)
        yield (24000, audio_array)


def create_stream(title: str):
    """Create a stream with the specified title"""
    return Stream(
        handler=ReplyOnPause(response, input_sample_rate=16000),
        modality="audio",
        mode="send-receive",
        ui_args={"title": title},
    )


# stream.ui.launch()
