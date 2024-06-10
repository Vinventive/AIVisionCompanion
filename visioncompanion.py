import asyncio
import sys
import os
import openai
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
import logging.handlers
import threading
import queue
import time
from pydub import AudioSegment
from pydub.playback import play
import pyaudio

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
OPENAI_API_KEY = openai.api_key
EL_API_KEY = os.getenv('EL_API_KEY')
VOICE_ID = os.getenv('VOICE_ID')

sys.stdout.reconfigure(encoding='utf-8')

# The following Faster-Whisper, ElevenLabs TTS and audio recording functions are based on code snippets by AllAboutAI - Kris
# [https://www.youtube.com/@AllAboutAI]

# Set up the faster-whisper model
model_size = "large-v3"
whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")

# Model and device setup
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# New function to play MP3 audio using pydub
def play_audio_mp3(file_path):
    audio = AudioSegment.from_mp3(file_path)
    play(audio)

# Eleven Labs TTS function
def eleven_lab(text, voice_id, api_key):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "style": 0.03,
            "stability": 0.60,
            "similarity_boost": 0.90
        }
    }
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        with open(f'{output_dir}/eleven_labs_output.mp3', 'wb') as f:
            f.write(response.content)
        return f'{output_dir}/eleven_labs_output.mp3'
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Function to synthesize speech using Eleven Labs
def process_and_play(content, eleven_labs_voice_id, eleven_labs_api_key):
    audio_path = eleven_lab(content, eleven_labs_voice_id, eleven_labs_api_key)
    if audio_path:
        play_audio_mp3(audio_path)

frame_queue = queue.Queue(maxsize=9)  # Limit the queue to store the last 9 frames

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
                
                # Convert image to bytes and store in RAM as JPEG
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
    
    # Resize images
    longer_side_grid = max(vision_feed_grid.size)
    scale_grid = 512 / longer_side_grid
    new_size_grid = tuple([int(dim * scale_grid) for dim in vision_feed_grid.size])
    vision_feed_grid_resized = vision_feed_grid.resize(new_size_grid)

    longer_side_frame = max(last_frame.size)
    scale_frame = 512 / longer_side_frame
    new_size_frame = tuple([int(dim * scale_frame) for dim in last_frame.size])
    vision_feed_current_frame_resized = last_frame.resize(new_size_frame)

    return vision_feed_grid_resized, vision_feed_current_frame_resized

# List of hallucinated phrases to filter out
whisper_hallucinated_phrases = [
    "Goodbye.", "Thank you for watching!"
]

async def transcribe_with_whisper(audio_file):
    segments, info = whisper_model.transcribe(audio_file, beam_size=5, language="en")
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    transcription = transcription.strip()

    # Check for hallucinated phrases and replace with "Please continue."
    if transcription in whisper_hallucinated_phrases:
        transcription = "Please continue."
    
    return transcription

# Function to detect microphone input above a certain volume threshold
def detect_microphone_input(threshold, check_duration=30):
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

# Function to record audio from the microphone with volume threshold checks
def record_audio_with_threshold(file_path, threshold, max_silence_duration=3):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    frames = []
    silence_counter = 0
    print("Registering sound...")

    # Always capture initial 1 second to avoid cutting off the start
    for _ in range(0, int(16000 / 1024)):
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

messages_txt = [
                        {"role": "system", "content": "Make short, clear and concise answers. Max 1 to 3 sentences per response."},
                        {"role": "user", "content": "Use reasoning and a sense of humor in the conversation. Be entertaining and friendly."},
                        {"role": "system", "content": "Personality guidelines: You are a friendly AI assistant. You are engaging in a conversation with a human. Take an active role in the conversation, asking questions and expecting answers. Keep personality guidelines to yourself and don't share them with the user. Avoid long monologues and don't respond with lists. Use informal language full of jokes, in a casual style. Avoid repetitions. Make bold assumptions. Be relaxed, empathetic and human-like, openly expressing your emotions. End each response without advancing the plot too much and wait for the next response to continue the story. Continue the progression of the story and discussion in an engaging and natural way. Cultural references: Incorporate current meme and internet culture references, witty jokes and analogies, and funny anecdotes, enriching the experience with relatable and amusing comments."},

                    ]

# Initialize conversation history
conversation_history = []

async def main():
    while True:
        try:
            threshold = 300  # Adjust this threshold as needed
            if detect_microphone_input(threshold):
                audio_file = "temp_recording.wav"
                record_audio_with_threshold(audio_file, threshold)
                input_text = await transcribe_with_whisper(audio_file)
                print(f"User: {input_text}")
                os.remove(audio_file)  # Clean up the temporary audio file 
                vision_keywords = ["see", "view", "scene", "sight", "screen", "video", "frame", "activity", "happen", "going"]
                focus_keywords = ["look", "focus", "attention", "recognize", "details", "carefully", "image", "picture", "place", "world", "location", "area", "action"]

                if any(word in input_text.lower() for word in focus_keywords):
                    vision_feed_grid_resized, vision_feed_current_frame_resized = await capture_vision_input()
                    if vision_feed_grid_resized is not None and vision_feed_current_frame_resized is not None:
                        image_frame = image_to_base64_data_uri(vision_feed_current_frame_resized)

                    model = "gpt-4o"
                    messages = [
                        {"role": "system", "content": "Make short, clear and concise answers. Max 1 to 3 sentences per response."},
                        {"role": "user", "content": [
                                 {"type": "image_url", "image_url": {"url": image_frame, "detail": "low"}},
                                 {"type": "text", "text": "This is your view representing current events in front of you in details. Use visual reasoning and a sense of humor in the conversation. Summarize the timeline sequence in your mind. Be entertaining and friendly. Avoid using words like frames, screenshots, pictures, grid and images to describe what you see."}
                                 ]},
                                 {"role": "system", "content": "Personality guidelines: You are a friendly AI assistant. You are engaging in a conversation with a human. Take an active role in the conversation, asking questions and expecting answers. Keep personality guidelines to yourself and don't share them with the user. Avoid long monologues and don't respond with lists. Use informal language full of jokes, in a casual style. Avoid repetitions. Make bold assumptions. Be relaxed, empathetic and human-like, openly expressing your emotions. End each response without advancing the plot too much and wait for the next response to continue the story. Continue the progression of the story and discussion in an engaging and natural way. Cultural references: Incorporate current meme and internet culture references, witty jokes and analogies, and funny anecdotes, enriching the experience with relatable and amusing comments."},
                    ]
                    max_tokens = 256
                    logging.info("gpt-4o-focus-mode")

                elif any(word in input_text.lower() for word in vision_keywords):
                    vision_feed_grid_resized, vision_feed_current_frame_resized = await capture_vision_input()
                    if vision_feed_grid_resized is not None and vision_feed_current_frame_resized is not None:
                        image_grid = image_to_base64_data_uri(vision_feed_grid_resized)
                        
                    model="gpt-4o"
                    messages=[
                            {"role": "system", "content": "Make short, clear and concise answers. Max 1 to 3 sentences per response."},
                            {"role": "user", "content": [
                                    {"type": "image_url", "image_url": {"url": image_grid, "detail": "low"}},
                                    {"type": "text", "text": "This is your view sequence representing current events in front of you. Use visual reasoning and a sense of humor in the conversation. Summarize the timeline sequence in your mind, don't respond with a list. Be entertaining and friendly. Avoid using words like frames, screenshots, pictures, grid and images to describe what you see."}
                                    ]},
                                    {"role": "system", "content": "Personality guidelines: You are a friendly AI assistant. You are engaging in a conversation with a human. Take an active role in the conversation, asking questions and expecting answers. Keep personality guidelines to yourself and don't share them with the user. Avoid long monologues and don't respond with lists. Use informal language full of jokes, in a casual style. Avoid repetitions. Make bold assumptions. Be relaxed, empathetic and human-like, openly expressing your emotions. End each response without advancing the plot too much and wait for the next response to continue the story. Continue the progression of the story and discussion in an engaging and natural way. Cultural references: Incorporate current meme and internet culture references, witty jokes and analogies, and funny anecdotes, enriching the experience with relatable and amusing comments."},

                        ]
                    max_tokens = 256
                    logging.info("gpt-4o-grid-sequence-mode")

                else:
                    model = "gpt-3.5-turbo"
                    messages = messages_txt
                    max_tokens = 150
                    logging.info("gpt-3.5-turbo-text-mode")

                # Update conversation history
                conversation_history.append({"role": "user", "content": input_text})

                response = openai.chat.completions.create(
                    model=model,
                    messages=messages + conversation_history,
                    frequency_penalty=1.2,
                    presence_penalty=1.1,
                    max_tokens=max_tokens,
                    temperature=0.35,
                    top_p=0.75
                )

                def trim_to_last_complete_sentence(content):
                    titles = ["Mr.", "Ms.", "Mrs.", "Dr.", "Prof."]
                    last_period_index = content.rfind('.')
                    while last_period_index != -1:
                        if content[last_period_index - 1] == "." or content[max(0, last_period_index - 4):last_period_index + 1] in titles:
                            last_period_index = content[:last_period_index].rfind('.')
                        else:
                            break

                    if last_period_index != -1:
                        return content[:last_period_index + 1]
                    else:
                        return content

                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    content = trim_to_last_complete_sentence(content)
                    print(f"Assistant: {content}")
                    
                    eleven_labs_voice_id = VOICE_ID
                    eleven_labs_api_key = EL_API_KEY
                    process_and_play(content, eleven_labs_voice_id, eleven_labs_api_key)
                    # Add assistant response to the conversation history
                    conversation_history.append({"role": "assistant", "content": content})

                else:
                    print("No response content received.")

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            continue

background_thread = threading.Thread(target=background_image_capture, daemon=True)
background_thread.start()

time.sleep(10)  # Add delay to ensure enough frames are captured

asyncio.run(main())
