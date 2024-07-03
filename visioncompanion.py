import asyncio
import sys
import os
import openai
import json
from PIL import Image
import wave
import torch
import base64
import io
import win32gui
import win32con
import mss
import pygetwindow as gw
import numpy as np
from dotenv import load_dotenv
import logging
import logging.handlers
import threading
import queue
import time
from pydub import AudioSegment
import pyaudio
import aiohttp
from groq import Groq
# logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
OPENAI_API_KEY = openai.api_key
EL_API_KEY = os.getenv('EL_API_KEY')
VOICE_ID = os.getenv('VOICE_ID')
GROQ_API = os.getenv('GROQ_API')

groq_client = Groq(api_key=GROQ_API, )

sys.stdout.reconfigure(encoding='utf-8')

audio_queue = queue.Queue()
playback_thread = None
playback_stop_event = threading.Event()
stop_recording_event = threading.Event()

def audio_playback_worker():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    output=True)
    
    first_chunk = True

    while not playback_stop_event.is_set():
        try:
            audio_segment = audio_queue.get(timeout=0.1)
            
            if first_chunk:
                time.sleep(0.1)  # Small delay before playback starts
                first_chunk = False

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
    if playback_thread is None or not playback_thread.is_alive():
        playback_stop_event.clear()
        playback_thread = threading.Thread(target=audio_playback_worker)
        playback_thread.start()

def stop_audio_playback():
    global playback_thread
    playback_stop_event.set()
    with audio_queue.mutex:
        audio_queue.queue.clear()
    if playback_thread:
        playback_thread.join()
    playback_thread = None

async def stream_eleven_labs(text, voice_id, api_key):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
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
                print(f"Error: {response.status}, {await response.text()}")
                return None

async def process_and_play_streaming(sentence, voice_id, api_key):
    audio_data = await stream_eleven_labs(sentence, voice_id, api_key)
    if audio_data:
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
        audio_segment = audio_segment.set_channels(1).set_frame_rate(44100)
        audio_queue.put(audio_segment)
        start_audio_playback_thread()

frame_queue = queue.Queue(maxsize=9)

def get_client_area(hwnd):
    rect = win32gui.GetClientRect(hwnd)
    point = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
    return (point[0], point[1], rect[2] - rect[0], rect[3] - rect[1])

window_title = input("Enter the title of the window: ")

logging.info("Loading frames. Please wait...")

def background_image_capture():
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

whisper_hallucinated_phrases = [
    "Goodbye.","Thanks for watching!", "Thank you for watching!", "I feel like I'm going to die.", "Thank you for watching."
]

async def transcribe_with_whisper(audio_data):
    # A dummy filename here since Groq API expects a filename
    dummy_filename = "audio.wav"
    
    transcription = groq_client.audio.transcriptions.create(
        file=(dummy_filename, audio_data),
        model="whisper-large-v3",
        response_format="text",
        language="en",
        temperature=0.0
    )
    
    transcription = transcription.strip()

    if transcription in whisper_hallucinated_phrases:
        transcription = "Please continue."
    
    return transcription

def detect_microphone_input(threshold, check_duration=20):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    print("Listening...")
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

def record_audio_with_threshold(threshold, max_silence_duration=1):
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

    # Instead of writing to a file, we'll use an in-memory buffer
    buffer = io.BytesIO()
    wf = wave.open(buffer, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Return the buffer's content
    return buffer.getvalue()

def image_to_base64_data_uri(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    base64_data = base64.b64encode(img_byte_arr).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_data}"

# Load and deserialize messages from .env
messages_txt = json.loads(os.getenv('MESSAGES_TXT'))
messages_focus_template = json.loads(os.getenv('MESSAGES_FOCUS_TEMPLATE'))
messages_grid_sequence_template = json.loads(os.getenv('MESSAGES_GRID_SEQUENCE_TEMPLATE'))

conversation_history = []

def trim_to_last_complete_sentence(content):
    last_period_index = content.rfind('.')
    if last_period_index != -1:
        return content[:last_period_index + 1]
    else:
        return content

vision_keywords = os.getenv("VISION_KEYWORDS").split(",")
focus_keywords = os.getenv("FOCUS_KEYWORDS").split(",")

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
    continuous_mode = True  # Start in continuous mode by default

    while True:
        try:
            threshold = 300
            stop_recording_event.clear()
            if detect_microphone_input(threshold):
                stop_audio_playback()  # Stop any ongoing playback
                audio_data = record_audio_with_threshold(threshold)
                stop_recording_event.set()  # Interrupt any ongoing recording
                input_text = await transcribe_with_whisper(audio_data)
                print(f"User: {input_text}")

                # Check for control keywords
                if "stop continuous mode" in input_text.lower():
                    continuous_mode = False
                    print("Continuous mode stopped. Waiting for user input.")
                    continue
                elif "start continuous mode" in input_text.lower():
                    continuous_mode = True
                    print("Continuous mode started.")
                    continue
                elif "exit vision companion" in input_text.lower():
                    print("Shutting down Vision Companion. Goodbye!")
                    return  # This will exit the main function and close the app
            elif continuous_mode:
                input_text = "Share what you think is happening."
                print(f"System: {input_text}")
            else:
                # If not in continuous mode and no input detected, wait for next iteration
                continue

            if any(word in input_text.lower() for word in focus_keywords):
                vision_feed_grid_resized, vision_feed_current_frame_resized = await capture_vision_input()
                if vision_feed_grid_resized is not None and vision_feed_current_frame_resized is not None:
                    image_frame = image_to_base64_data_uri(vision_feed_current_frame_resized)
                    messages_focus = [message.copy() for message in messages_focus_template]
                    messages_focus[1]['content'][0]['image_url']['url'] = image_frame

                model = "gpt-4o"
                messages = messages_focus
                max_tokens = 256
                logging.info("gpt-4o-focus-mode")

            elif any(word in input_text.lower() for word in vision_keywords):
                vision_feed_grid_resized, vision_feed_current_frame_resized = await capture_vision_input()
                if vision_feed_grid_resized is not None and vision_feed_current_frame_resized is not None:
                    image_grid = image_to_base64_data_uri(vision_feed_grid_resized)
                    messages_grid_sequence = [message.copy() for message in messages_grid_sequence_template]
                    messages_grid_sequence[1]['content'][0]['image_url']['url'] = image_grid
                
                model = "gpt-4o"
                messages = messages_grid_sequence
                max_tokens = 256
                logging.info("gpt-4o-grid-sequence-mode")

            else:
                model = "gpt-3.5-turbo"
                messages = messages_txt
                max_tokens = 150
                logging.info("gpt-3.5-turbo-text-mode")

            conversation_history.append({"role": "user", "content": input_text})

            full_response = ""
            sentence_buffer = ""
            async for sentence in stream_openai_response(client, model, messages + conversation_history, max_tokens, 0.35, 0.75, 1.2, 1.1):
                full_response += sentence + " "
                sentence_buffer += sentence + " "
                
                # Process and queue audio for complete sentences
                if sentence.strip().endswith(('.', '!', '?')):
                    print(f"Assistant: {sentence_buffer.strip()}")
                    await process_and_play_streaming(sentence_buffer.strip(), VOICE_ID, EL_API_KEY)
                    sentence_buffer = ""

            # Process any remaining text in the buffer
            if sentence_buffer.strip():
                print(f"Assistant: {sentence_buffer.strip()}")
                await process_and_play_streaming(sentence_buffer.strip(), VOICE_ID, EL_API_KEY)

            conversation_history.append({"role": "assistant", "content": full_response.strip()})

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            continue


# Start the background image capture thread
background_thread = threading.Thread(target=background_image_capture, daemon=True)
background_thread.start()

# Wait for initial frames to be captured
time.sleep(10)

# Run the main asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())
