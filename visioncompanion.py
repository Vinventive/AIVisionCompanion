import asyncio
import sys
import os
import openai
import json
from faster_whisper import WhisperModel
from PIL import Image
import wave
import torch
import base64
import io
import requests
import win32gui
import win32con
import mss
import pygetwindow as gw
import numpy as np
from dotenv import load_dotenv
import logging
import threading
import queue
import time
from pydub import AudioSegment
import pyaudio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Set up API keys and constants
openai.api_key = os.getenv('OPENAI_API_KEY')
OPENAI_API_KEY = openai.api_key
EL_API_KEY = os.getenv('EL_API_KEY')
VOICE_ID = os.getenv('VOICE_ID')

# Configure system settings
sys.stdout.reconfigure(encoding='utf-8')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Whisper model
model_size = "large-v3"
whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")

# Set up output directory
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Initialize queues and events
audio_queue = queue.Queue()
frame_queue = queue.Queue(maxsize=9)
playback_stop_event = threading.Event()
stop_recording_event = threading.Event()

# Load conversation templates and keywords
messages_focus_template = json.loads(os.getenv('MESSAGES_FOCUS_TEMPLATE'))
messages_grid_sequence_template = json.loads(os.getenv('MESSAGES_GRID_SEQUENCE_TEMPLATE'))
vision_keywords = os.getenv("VISION_KEYWORDS").split(",")
focus_keywords = os.getenv("FOCUS_KEYWORDS").split(",")

# Initialize conversation history
conversation_history = []

# Whisper hallucination phrases
whisper_hallucinated_phrases = [
    "Goodbye.", "Thanks for watching!", "Thank you for watching!", 
    "I feel like I'm going to die.", "Thank you for watching."
]

def audio_playback_worker():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, output=True)
    
    while not playback_stop_event.is_set():
        try:
            audio_segment = audio_queue.get(timeout=0.1)
            chunk_size = 1024
            for i in range(0, len(audio_segment), chunk_size):
                if playback_stop_event.is_set():
                    break
                chunk = audio_segment[i:i+chunk_size].raw_data
                stream.write(chunk)
            audio_queue.task_done()
        except queue.Empty:
            continue

    stream.stop_stream()
    stream.close()
    p.terminate()

def start_audio_playback_thread():
    global playback_thread
    if not hasattr(start_audio_playback_thread, 'playback_thread') or not start_audio_playback_thread.playback_thread.is_alive():
        playback_stop_event.clear()
        start_audio_playback_thread.playback_thread = threading.Thread(target=audio_playback_worker)
        start_audio_playback_thread.playback_thread.start()

def stop_audio_playback():
    playback_stop_event.set()
    with audio_queue.mutex:
        audio_queue.queue.clear()
    if hasattr(start_audio_playback_thread, 'playback_thread'):
        start_audio_playback_thread.playback_thread.join()

async def stream_eleven_labs(text, voice_id, api_key):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {
            "style": 0.03,
            "stability": 0.60,
            "similarity_boost": 0.90
        }
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status == 200:
                return await response.read()
            else:
                logging.error(f"ElevenLabs API error: {response.status}, {await response.text()}")
                return None

async def process_and_play_streaming(sentence, voice_id, api_key):
    audio_data = await stream_eleven_labs(sentence, voice_id, api_key)
    if audio_data:
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
        audio_segment = audio_segment.set_channels(1).set_frame_rate(44100)
        audio_queue.put(audio_segment)
        start_audio_playback_thread()

def get_client_area(hwnd):
    rect = win32gui.GetClientRect(hwnd)
    point = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
    return (point[0], point[1], rect[2] - rect[0], rect[3] - rect[1])

def background_image_capture(window_title):
    while True:
        try:
            window = gw.getWindowsWithTitle(window_title)
            if len(window) == 0:
                logging.info(f"No window with title '{window_title}' found.")
                time.sleep(1)
                continue

            window = window[0]
            hwnd = window._hWnd

            placement = win32gui.GetWindowPlacement(hwnd)
            if placement[1] != win32con.SW_SHOWMAXIMIZED:
                win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
                time.sleep(0.5)

            x, y, width, height = get_client_area(hwnd)

            with mss.mss() as sct:
                window_size = {"top": y, "left": x, "width": width, "height": height}
                vision_input = sct.grab(window_size)

                img_bytes = mss.tools.to_png(vision_input.rgb, vision_input.size)
                vision_feed = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                
                img_byte_arr = io.BytesIO()
                vision_feed.save(img_byte_arr, format='JPEG')
                vision_bytes = img_byte_arr.getvalue()

            if frame_queue.full():
                frame_queue.get()
            frame_queue.put(vision_bytes)

            time.sleep(1)

        except Exception as e:
            logging.error(f"Error in background image capture: {e}")
            time.sleep(1)

def combine_frames_to_grid():
    frames = list(frame_queue.queue)
    if len(frames) < 9:
        return None

    images = [Image.open(io.BytesIO(frame)) for frame in frames]
    frame_size = images[0].size
    grid_size = (frame_size[0] * 3, frame_size[1] * 3)
    grid_image = Image.new('RGB', grid_size)

    for i in range(3):
        for j in range(3):
            grid_image.paste(images[i * 3 + j], (j * frame_size[0], i * frame_size[1]))

    return grid_image

async def capture_vision_input():
    if frame_queue.qsize() < 9:
        logging.info("Not enough frames captured yet.")
        return None, None

    last_frame = Image.open(io.BytesIO(frame_queue.queue[-1]))
    vision_feed_grid = combine_frames_to_grid()
    if vision_feed_grid is None:
        return None, None
    
    longer_side_grid = max(vision_feed_grid.size)
    scale_grid = 512 / longer_side_grid
    new_size_grid = tuple([int(dim * scale_grid) for dim in vision_feed_grid.size])
    vision_feed_grid_resized = vision_feed_grid.resize(new_size_grid)

    longer_side_frame = max(last_frame.size)
    scale_frame = 512 / longer_side_frame
    new_size_frame = tuple([int(dim * scale_frame) for dim in last_frame.size])
    vision_feed_current_frame_resized = last_frame.resize(new_size_frame)

    return vision_feed_grid_resized, vision_feed_current_frame_resized

async def transcribe_with_whisper(audio_file):
    segments, info = whisper_model.transcribe(audio_file, beam_size=5, language="en")
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    transcription = transcription.strip()

    if transcription in whisper_hallucinated_phrases:
        transcription = "Please continue."
    
    return transcription

def detect_microphone_input(threshold, check_duration=30):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    
    sys.stdout.write("\rListening... ")
    sys.stdout.flush()
    
    for _ in range(0, int(16000 / 1024 * check_duration)):
        data = stream.read(1024)
        audio_data = np.frombuffer(data, np.int16)
        volume = np.linalg.norm(audio_data) / np.sqrt(len(audio_data))
        if volume > threshold:
            stream.stop_stream()
            stream.close()
            p.terminate()
            return True
    stream.stop_stream()
    stream.close()
    p.terminate()
    return False

def record_audio_with_threshold(file_path, threshold, max_silence_duration=2):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    frames = []
    silence_counter = 0
    print("Registering sound...")

    # Always capture initial 2 seconds to avoid cutting off the start
    for _ in range(0, int(16000 / 1024 * 2)):
        data = stream.read(1024)
        frames.append(data)

    while True:
        data = stream.read(1024)
        frames.append(data)
        audio_data = np.frombuffer(data, np.int16)
        volume = np.linalg.norm(audio_data) / np.sqrt(len(audio_data))
        
        if volume < threshold:
            silence_counter += 1
        else:
            silence_counter = 0
        
        if silence_counter > int(16000 / 1024 * max_silence_duration):
            break

    print("Done.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def image_to_base64_data_uri(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    base64_data = base64.b64encode(img_byte_arr).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_data}"

def trim_conversation_history(max_words):
    global conversation_history
    total_words = sum(len(message['content'].split()) for message in conversation_history)
    while total_words > max_words and len(conversation_history) > 2:
        removed_message = conversation_history.pop(0)
        total_words -= len(removed_message['content'].split())

def determine_model_and_messages(input_text):
    if any(word in input_text.lower() for word in focus_keywords):
        model = "gpt-4o"
        messages = messages_focus_template
        max_tokens = 256
        logging.info("gpt-4o-focus-mode")
    elif any(word in input_text.lower() for word in vision_keywords):
        model = "gpt-4o"
        messages = messages_grid_sequence_template
        max_tokens = 256
        logging.info("gpt-4o-grid-sequence-mode")
    else:
        model = "gpt-4o-mini"
        messages = messages_focus_template
        max_tokens = 150
        logging.info("gpt-4o-continious-mode")
    return model, messages, max_tokens

async def prepare_messages(messages_template):
    vision_feed_grid_resized, vision_feed_current_frame_resized = await capture_vision_input()
    messages = [message.copy() for message in messages_template]
    
    if messages_template == messages_focus_template and vision_feed_current_frame_resized is not None:
        image_frame = image_to_base64_data_uri(vision_feed_current_frame_resized)
        messages[1]['content'][0]['image_url']['url'] = image_frame
    elif messages_template == messages_grid_sequence_template and vision_feed_grid_resized is not None:
        image_grid = image_to_base64_data_uri(vision_feed_grid_resized)
        messages[1]['content'][0]['image_url']['url'] = image_grid
    
    return messages

async def stream_openai_response(client, model, messages, max_tokens, temperature, top_p, frequency_penalty, presence_penalty):
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stream=True
    )

    full_response = ""
    current_sentence = ""

    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            current_sentence += content

            if content.endswith(('.', '!', '?')):
                yield current_sentence.strip()
                current_sentence = ""

    if current_sentence:
        yield current_sentence.strip()

async def main():
    global stop_recording_event
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    # Create a ThreadPoolExecutor for audio processing
    audio_executor = ThreadPoolExecutor(max_workers=2)

    # Get the window title from user input
    window_title = input("Enter the title of the window: ")

    # Start the background image capture thread
    background_thread = threading.Thread(target=background_image_capture, args=(window_title,), daemon=True)
    background_thread.start()

    # Wait for initial frames to be captured
    logging.info("Loading frames. Please wait...")
    await asyncio.sleep(10)

    while True:
        try:
            threshold = 500
            stop_recording_event.clear()
            
            # Run microphone input detection in the thread pool
            loop = asyncio.get_event_loop()
            if await loop.run_in_executor(audio_executor, detect_microphone_input, threshold):
                stop_audio_playback()
                
                # Run audio recording in the thread pool
                audio_file = "temp_recording.wav"
                await loop.run_in_executor(audio_executor, record_audio_with_threshold, audio_file, threshold)
                
                stop_recording_event.set()
                
                # Transcribe audio (this is already async)
                input_text = await transcribe_with_whisper(audio_file)
                print(f"User: {input_text}")
                await loop.run_in_executor(audio_executor, os.remove, audio_file)

                # Determine the model and prepare messages
                model, messages_template, max_tokens = determine_model_and_messages(input_text)
                messages = await prepare_messages(messages_template)

                # Trim conversation history if it exceeds 100000 words
                trim_conversation_history(100000)

                conversation_history.append({"role": "user", "content": input_text})

                # Process OpenAI response and ElevenLabs TTS in parallel
                full_response = ""
                async for sentence in stream_openai_response(client, model, messages + conversation_history, max_tokens, 0.35, 0.75, 1.2, 1.1):
                    print(f"Assistant: {sentence}")
                    full_response += sentence + " "
                    
                    # Start ElevenLabs API call immediately for each sentence
                    await process_and_play_streaming(sentence, VOICE_ID, EL_API_KEY)

                # Append the full response to conversation history
                conversation_history.append({"role": "assistant", "content": full_response.strip()})

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            continue

if __name__ == "__main__":
    asyncio.run(main())
