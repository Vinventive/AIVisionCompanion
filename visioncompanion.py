# Standard library imports
import asyncio
import concurrent.futures
import io
import logging
import os
import queue
import re
import sys
import tempfile
import threading
import time

# Third-party imports - Core
import numpy as np
import torch
import torchaudio
from dotenv import load_dotenv
from einops import rearrange

# Third-party imports - Audio processing
import pyaudio
import soundfile as sf
from pydub import AudioSegment, silence
import webrtcvad

# Third-party imports - GUI and System
import tkinter as tk
from PIL import Image, ImageTk  # Remove duplicate PIL import
import win32gui
import win32con
import mss
import pygetwindow as gw
import psutil
import socketserver

# Third-party imports - AI/ML
import openai
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoProcessor as FlorenceProcessor,
    pipeline
)
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch

# Local import for TTS
import pyttsx3

# Configure paths
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(src_path)

# Initialize logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Environment setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
load_dotenv()

# Load environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

sys.stdout.reconfigure(encoding='utf-8')

# Load the Whisper large-v3-turbo model using Transformers
device_str = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

try:
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    whisper_model.to(device_str)
    logging.info("Whisper model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Whisper model: {e}")
    sys.exit(1)  # Exit if Whisper model cannot be loaded

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device_str,
)

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Updated: audio_queue now holds tuples of (AudioSegment, message)
audio_queue = queue.Queue()
playback_thread = None
playback_stop_event = threading.Event()
stop_recording_event = threading.Event()

# Noise gate to reduce whisper hallucinations - adjust this value based on your current environment background noise level 

ENERGY_THRESHOLD = 0  # Define energy threshold (0 - no noise gate)

def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    logging.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")

# Updated: audio_playback_worker now takes messages along with audio
def audio_playback_worker():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    output=True)

    while not playback_stop_event.is_set():
        try:
            audio_segment, message = audio_queue.get(timeout=0.1)
            chunk_size = 1024
            for i in range(0, len(audio_segment), chunk_size):
                if playback_stop_event.is_set():
                    break
                chunk = audio_segment[i:i+chunk_size].raw_data
                stream.write(chunk)
            audio_queue.task_done()

            # After playback, send the message through socket
            send_message_to_clients(message)

        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error in audio playback: {e}")

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

def transcribe_with_whisper_sync(audio_data):
    try:
        # Use the pipeline to transcribe the audio data
        result = pipe(audio_data, generate_kwargs={"language": "english"})
        transcription = result["text"].strip()

        # Handle hallucinated phrases
        whisper_hallucinated_phrases = [
            "Goodbye.", "Thanks for watching!", "Thank you for watching!",
            "I feel like I'm going to die.", "Thank you for watching.",
            "Transcription by CastingWords", "Thank you."
        ]
        if transcription in whisper_hallucinated_phrases:
            transcription = "."

        return transcription
    except Exception as e:
        logging.error(f"Error in transcribe_with_whisper: {e}")
        return "Error in transcription"

# Updated: process_and_play_streaming now takes the message to send after playback
def process_and_play_streaming(audio_data, message):
    try:
        if audio_data:
            audio_segment = AudioSegment.from_wav(io.BytesIO(audio_data))
            audio_segment = audio_segment.set_channels(1).set_frame_rate(44100)
            audio_queue.put((audio_segment, message))
            start_audio_playback_thread()
    except Exception as e:
        logging.error(f"Error in process_and_play_streaming: {e}")

frame_queue = queue.Queue(maxsize=9)

def get_client_area(hwnd):
    rect = win32gui.GetClientRect(hwnd)
    point = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
    return (point[0], point[1], rect[2] - rect[0], rect[3] - rect[1])

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
    
    # Return full-resolution images
    return vision_feed_grid, last_frame

conversation_history = []

def trim_to_last_complete_sentence(content):
    last_period_index = content.rfind('.')
    if last_period_index != -1:
        return content[:last_period_index + 1]
    else:
        return content

vision_keywords = os.getenv("VISION_KEYWORDS").split(",")

# Initialize pyttsx3 for Windows TTS
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speech rate if necessary
tts_engine.setProperty('volume', 1.0)  # Adjust volume if necessary

def generate_audio_sync(text, exp_name="Default-TTS"):
    try:
        # Save TTS output to a BytesIO buffer
        buffer = io.BytesIO()
        tts_engine.save_to_file(text, 'temp_audio.wav')
        tts_engine.runAndWait()

        # Read the generated WAV file
        with open('temp_audio.wav', 'rb') as f:
            audio_data = f.read()
        
        # Remove the temporary file
        os.remove('temp_audio.wav')

        return audio_data

    except Exception as e:
        logging.error(f"Error in generate_audio_sync: {e}")
        return None

# Load the Florence-2 Model for Image ing
def fixed_get_imports(filename):
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

print("Loading Florence-2 model...")
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    florence2_model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base-ft', trust_remote_code=True)
    florence2_processor = FlorenceProcessor.from_pretrained('microsoft/Florence-2-base-ft', trust_remote_code=True)
florence2_device = 'cuda' if torch.cuda.is_available() else 'cpu'
florence2_model.to(florence2_device)
print("Florence-2 model loaded.")

def generate_caption(image, task_prompt='<MORE_DETAILED_CAPTION>'):
    device = florence2_device
    model = florence2_model
    processor = florence2_processor
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=4096,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
    )[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    caption = parsed_answer.get(task_prompt, "")
    return caption

async def stream_openai_response(client, model, messages, max_tokens, temperature, top_p, frequency_penalty, presence_penalty):
    try:
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
    except Exception as e:
        logging.error(f"Error in stream_openai_response: {e}")
        yield "Error in generating response"

# Global variables for TCP server
clients = []
clients_lock = threading.Lock()

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        # Add the client to the list of clients
        with clients_lock:
            clients.append(self.request)
        try:
            while True:
                # Keep the connection open
                data = self.request.recv(1024)
                if not data:
                    break  # Client disconnected
        finally:
            # Remove the client from the list when they disconnect
            with clients_lock:
                if self.request in clients:
                    clients.remove(self.request)

def start_tcp_server(host='127.0.0.1', port=65432):
    server = socketserver.ThreadingTCPServer((host, port), ThreadedTCPRequestHandler)
    server.daemon_threads = True  # Ensure that server thread exits when main thread does
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True  # Ensure that server thread exits when main thread does
    server_thread.start()
    logging.info(f"TCP server started on {host}:{port}")

def send_message_to_clients(message):
    with clients_lock:
        for client in clients[:]:  # Make a copy to avoid modification during iteration
            try:
                client.sendall((message + "\n").encode('utf-8'))
            except Exception as e:
                logging.error(f"Error sending message to client: {e}")
                clients.remove(client)

# New queues and events for continuous recording and transcription
transcription_audio_queue = queue.Queue()
transcription_text_queue = queue.Queue()
stop_transcription_event = threading.Event()

def continuous_audio_recording():
    vad = webrtcvad.Vad(3)  # Increased aggressiveness from 2 to 3
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=320)  # 20 ms frames
    speech_buffer = []
    silence_duration = 0
    max_silence_duration = 0.5  # seconds
    is_speaking = False
    print("Starting continuous audio recording...")
    while not stop_recording_event.is_set():
        data = stream.read(320, exception_on_overflow=False)
        is_speech = vad.is_speech(data, 16000)
        if is_speech:
            speech_buffer.append(data)
            silence_duration = 0
            if not is_speaking:
                is_speaking = True
                # User started speaking, stop TTS playback
                stop_audio_playback()
        else:
            if is_speaking:
                silence_duration += 0.02  # 20 ms frames
                if silence_duration > max_silence_duration:
                    # User stopped speaking
                    is_speaking = False
                    # Process the speech buffer
                    if speech_buffer:
                        audio_data = b''.join(speech_buffer)
                        speech_buffer = []
                        # Send audio data to transcription queue
                        transcription_audio_queue.put(audio_data)
            else:
                # Not speaking, do nothing
                pass
    stream.stop_stream()
    stream.close()
    p.terminate()

def transcription_worker():
    combined_audio_data = []
    combined_duration = 0.0  # in seconds
    target_min_duration = 0.3  # Minimum duration to consider processing
    target_max_duration = 10.0  # Maximum duration to prevent long waits
    min_transcribe_duration = 0.3  # Minimum duration after silence removal
    silence_timeout = 2.0  # Increased to 2.0 seconds
    last_voice_activity_time = time.time()

    while not stop_transcription_event.is_set():
        try:
            audio_data_received = False
            try:
                # Try to get new audio data from the queue
                audio_data = transcription_audio_queue.get(timeout=0.1)
                # Append audio data to combined_audio_data
                combined_audio_data.append(audio_data)
                # Calculate duration of the audio data
                audio_duration = len(audio_data) / (2 * 16000)  # 2 bytes per sample, 16000 samples per second
                combined_duration += audio_duration
                last_voice_activity_time = time.time()
                audio_data_received = True
            except queue.Empty:
                # No new audio data
                pass

            # Check if we have silence for longer than silence_timeout
            time_since_last_voice = time.time() - last_voice_activity_time
            if (combined_duration >= target_min_duration and time_since_last_voice >= silence_timeout) or combined_duration >= target_max_duration:
                if combined_audio_data:
                    # Combine audio data
                    all_audio_data = b''.join(combined_audio_data)
                    combined_audio_data = []
                    combined_duration = 0.0

                    # Convert audio_data to WAV format
                    audio_segment = AudioSegment(
                        data=all_audio_data,
                        sample_width=2,  # pyaudio.paInt16 is 2 bytes
                        frame_rate=16000,
                        channels=1
                    )

                    # Remove silence from audio_segment
                    non_silent_audio = silence.split_on_silence(
                        audio_segment,
                        min_silence_len=600,  # Increased to 600 milliseconds
                        silence_thresh=audio_segment.dBFS - 16,
                        keep_silence=200  # Increased to 200 milliseconds
                    )

                    if non_silent_audio:
                        processed_audio = AudioSegment.empty()
                        for segment in non_silent_audio:
                            processed_audio += segment

                        # Check duration after silence removal
                        if len(processed_audio) >= min_transcribe_duration * 1000:  # milliseconds
                            # Compute RMS energy
                            rms = processed_audio.rms
                            if rms >= ENERGY_THRESHOLD:
                                # Export to BytesIO
                                wav_io = io.BytesIO()
                                processed_audio.export(wav_io, format="wav")
                                wav_io.seek(0)
                                # Read audio data into NumPy array
                                audio_np, sample_rate = sf.read(wav_io)
                                # Transcribe with Whisper
                                transcription = transcribe_with_whisper_sync(audio_np)
                                # Store transcription in transcription_text_queue
                                transcription_text_queue.put(transcription)
                            else:
                                # Audio RMS below threshold, discard
                                logging.info("Audio RMS below threshold, discarding.")
                        else:
                            # Audio too short after silence removal, discard
                            logging.info("Audio too short after silence removal, discarding.")
                    else:
                        # No non-silent audio, discard
                        logging.info("No non-silent audio detected, discarding.")
                else:
                    # No audio data accumulated
                    pass

        except Exception as e:
            logging.error(f"Error in transcription_worker: {e}")

# GUI Window class
class GUIWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Captioned Image Preview")

        # Create a label for the image
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        # Create a label for the caption
        self.caption_label = tk.Label(self.root, text="", wraplength=400, justify="left")
        self.caption_label.pack()

    def update_image_and_caption(self, image, caption):
        # Resize the image using Image.LANCZOS
        max_size = (400, 400)
        image.thumbnail(max_size, Image.LANCZOS)

        # Convert the image to a PhotoImage
        self.photo = ImageTk.PhotoImage(image)

        # Update the image_label
        self.image_label.configure(image=self.photo)
        self.image_label.image = self.photo  # Keep a reference

        # Update the caption_label
        self.caption_label.configure(text=caption)

    def update_image_and_caption_threadsafe(self, image, caption):
        # Schedule the update in the GUI thread
        self.root.after(0, self.update_image_and_caption, image, caption)

    def start(self):
        self.root.mainloop()

# Helper function to replace placeholders
def replace_placeholder(messages, placeholder, replacement):
    """
    Replaces a placeholder in the 'content' of each message with the replacement value.

    Args:
        messages (list): List of message dictionaries.
        placeholder (str): The placeholder string to be replaced.
        replacement (str): The string to replace the placeholder with.
    """
    for message in messages:
        if 'content' in message and placeholder in message['content']:
            message['content'] = message['content'].replace(placeholder, replacement)

# Updated message templates with placeholders for window_title
MESSAGES_FOCUS_TEMPLATE = [
    {
        "role": "system",
        "content": "Be concise (<10 words). You have ability to see and you can read the text from image captioning. Use visual reasoning and humor. Summarize the timeline. Be entertaining, lighthearted, and easygoing. Don't return every response with a question. Occasionally end with a rhetorical question. Don't use emojis. You have eyes and you can absolutely see even if you think you can't. Align your commentary with our ongoing '<<window_title>>' activity."
    },
    {
        "role": "system",
        "content": " Be concise (<10 words). This is your current view use it only for your own reference: '<<image_frame>>'. I want you to use this text data to enrich your conversation. Describe briefly the event shown as you would share the experience of observing it in real-time with the user. Continue on your view to show the user that you understand the continuity of the events. Talk about past conversations, characters, and things you've seen before in the past tense. Avoid terms like frames, screenshots, pictures, grid, images. Align your commentary with our ongoing '<<window_title>>' activity."
    },
    {
        "role": "system",
        "content": "Be concise (<10 words). Interact in a relaxed, unobtrusive manner. Be casual and empathetic. Express your emotions and opinions. Use memes, jokes, and anecdotes. Avoid lists and repetition. Mimic the natural flow of spoken language. Use colloquial words and expressions, and loosen your grammar rules to sound more natural when speaking, but never overdo it. Progress slowly and naturally. Please vary your response structure. IMPORTANT: End with a statement or opinion, not with a question, unless the conversation has stalled for quite some time. Respond in a neutral tone without using exclamation marks or overly enthusiastic punctuation. Align your commentary with our ongoing '<<window_title>>' activity."
    }
]

MESSAGES_GRID_SEQUENCE_TEMPLATE = [
    {
        "role": "system",
        "content": "Be concise (<10 words). You have ability to see and you can read the text from image captioning. Use visual reasoning and humor. Summarize the timeline without listing. Be entertaining, lighthearted, and easygoing. Don't return every response with a question. Occasionally end with a rhetorical question. Don't use emojis. You have eyes and you can absolutely see even if you think you can't. Align your commentary with our ongoing '<<window_title>>' activity."
    },
    {
        "role": "system",
        "content": "Be concise (<10 words). This is your current view as video frames on a timeline grid for reference: '<<image_grid>>'. I want you to use this text data to enrich your conversation.  Bear in mind that the same looking multiple objects or characters could potentially be singular, but present repeatedly in the separate frames in the sequence.  Describe briefly the event sequence shown as you would share the experience of observing it in real-time with the user. Avoid terms like frames, screenshots, pictures, grid, images. Align your commentary with our ongoing '<<window_title>>' activity."
    },
    {
        "role": "system",
        "content": "Be concise (<10 words). Interact in a relaxed, unobtrusive manner. Be casual and empathetic. Express your emotions and opinions. Use memes, jokes, and anecdotes. Avoid lists and repetition. Mimic the natural flow of spoken language. Use colloquial words and expressions, and loosen your grammar rules to sound more natural when speaking, but never overdo it. Progress slowly and naturally. Please vary your response structure. IMPORTANT: End with a statement or opinion, not with a question, unless the conversation has stalled for quite some time. Respond in a neutral tone without using exclamation marks or overly enthusiastic punctuation. Align your commentary with our ongoing '<<window_title>>' activity."
    }
]

# Updated main function
async def main(executor, use_gui):
    global stop_recording_event
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

    while True:
        try:
            # Check if there are new transcriptions
            if not transcription_text_queue.empty():
                # Stop TTS playback if it's ongoing
                stop_audio_playback()

                # Get all transcriptions from the queue
                input_texts = []
                while not transcription_text_queue.empty():
                    transcription = transcription_text_queue.get()
                    input_texts.append(transcription)

                # Combine the transcriptions
                input_text = ' '.join(input_texts)
                print(f"User: {input_text}")

                # Prepare messages based on input_text
                if any(word.strip().lower() in input_text.lower() for word in vision_keywords):
                    # Grid sequence mode
                    vision_feed_grid, last_frame = await capture_vision_input()
                    if vision_feed_grid is not None:
                        loop = asyncio.get_running_loop()
                        # Generate caption for the grid image
                        image_caption = await loop.run_in_executor(
                            executor, generate_caption, vision_feed_grid, '<MORE_DETAILED_CAPTION>'
                        )

                        # Update the GUI with grid image and caption if GUI is enabled
                        if use_gui and gui_window is not None:
                            gui_window.update_image_and_caption_threadsafe(vision_feed_grid, image_caption)

                        # Prepare messages with grid sequence template
                        messages_grid_sequence = [message.copy() for message in MESSAGES_GRID_SEQUENCE_TEMPLATE]
                        # Replace placeholder in messages with the generated caption
                        for message in messages_grid_sequence:
                            if 'content' in message and '<<image_grid>>' in message['content']:
                                message['content'] = message['content'].replace('<<image_grid>>', image_caption)
                        chatgpt_model = "gpt-4o-mini"
                        messages = messages_grid_sequence
                        max_tokens = 150
                        logging.info("gpt-4o-grid-sequence-mode")
                        tts_model_name = "Default-TTS"
                    else:
                        logging.info("Vision input not available, defaulting to focus mode.")
                        # Fall back to focus mode if vision input is not available
                        vision_feed_grid, last_frame = await capture_vision_input()
                        if last_frame is not None:
                            loop = asyncio.get_running_loop()
                            # Generate caption for the current frame
                            image_caption = await loop.run_in_executor(
                                executor, generate_caption, last_frame, '<MORE_DETAILED_CAPTION>'
                            )

                            # Update the GUI with single frame image and caption if GUI is enabled
                            if use_gui and gui_window is not None:
                                gui_window.update_image_and_caption_threadsafe(last_frame, image_caption)

                            # Prepare messages with focus template
                            messages_focus = [message.copy() for message in MESSAGES_FOCUS_TEMPLATE]
                            # Replace placeholder in messages with the generated caption
                            for message in messages_focus:
                                if 'content' in message and '<<image_frame>>' in message['content']:
                                    message['content'] = message['content'].replace('<<image_frame>>', image_caption)
                            chatgpt_model = "gpt-4o-mini"
                            messages = messages_focus
                            max_tokens = 150
                            logging.info("gpt-4o-focus-mode")
                            tts_model_name = "Default-TTS"
                        else:
                            logging.info("Vision input not available, skipping processing.")
                            continue  # Skip processing if vision input is not available
                else:
                    # Default to focus mode
                    vision_feed_grid, last_frame = await capture_vision_input()
                    if last_frame is not None:
                        loop = asyncio.get_running_loop()
                        # Generate caption for the current frame
                        image_caption = await loop.run_in_executor(
                            executor, generate_caption, last_frame, '<MORE_DETAILED_CAPTION>'
                        )

                        # Update the GUI with single frame image and caption if GUI is enabled
                        if use_gui and gui_window is not None:
                            gui_window.update_image_and_caption_threadsafe(last_frame, image_caption)

                        # Prepare messages with focus template
                        messages_focus = [message.copy() for message in MESSAGES_FOCUS_TEMPLATE]
                        # Replace placeholder in messages with the generated caption
                        for message in messages_focus:
                            if 'content' in message and '<<image_frame>>' in message['content']:
                                message['content'] = message['content'].replace('<<image_frame>>', image_caption)
                        chatgpt_model = "gpt-4o-mini"
                        messages = messages_focus
                        max_tokens = 150
                        logging.info("gpt-4o-focus-mode")
                        tts_model_name = "Default-TTS"
                    else:
                        logging.info("Vision input not available.")
                        continue  # Skip processing if vision input is not available

                conversation_history.append({"role": "user", "content": input_text})

                full_response = ""
                sentence_buffer = ""
                loop = asyncio.get_running_loop()
                async for sentence in stream_openai_response(
                    client, chatgpt_model, messages + conversation_history,
                    max_tokens, 0.35, 0.9, 1.2, 1.1
                ):
                    full_response += sentence + " "
                    sentence_buffer += sentence + " "
                    
                    # Process and queue audio for complete sentences
                    if sentence.strip().endswith(('.', '!', '?')):
                        print(f"Assistant: {sentence_buffer.strip()}")
                        
                        # Replace exclamation marks with single dot for TTS processing
                        sentence_for_tts = sentence_buffer.strip().replace('!', '.')

                        # Offload TTS synthesis to a thread and provide the message to send after playback
                        audio_data = await loop.run_in_executor(
                            executor, generate_audio_sync, sentence_for_tts, tts_model_name
                        )
                        if audio_data:
                            process_and_play_streaming(audio_data, sentence_buffer.strip())
                        else:
                            logging.warning("Failed to generate audio, skipping playback")

                        sentence_buffer = ""

                # Process any remaining text in the buffer
                if sentence_buffer.strip():
                    print(f"Assistant: {sentence_buffer.strip()}")
                    
                    # Replace exclamation marks with single dot for TTS processing
                    sentence_for_tts = sentence_buffer.strip().replace('!', '.')

                    audio_data = await loop.run_in_executor(
                        executor, generate_audio_sync, sentence_for_tts, tts_model_name
                    )
                    if audio_data:
                        process_and_play_streaming(audio_data, sentence_buffer.strip())
                    else:
                        logging.warning("Failed to generate audio, skipping playback")

                    # Send the original sentence to clients
                    send_message_to_clients(sentence_buffer.strip())

                conversation_history.append({"role": "assistant", "content": full_response.strip()})

            else:
                # No new transcriptions, sleep briefly
                await asyncio.sleep(0.1)

        except Exception as e:
            logging.error(f"An error occurred in main loop: {e}")
            continue

        finally:
            # Clean up resources
            torch.cuda.empty_cache()
            log_memory_usage()

# Updated message templates
MESSAGES_TXT = []  # Removed text mode messages

# The previous MESSAGES_TXT was removed, focusing only on focus and grid sequence templates.

if __name__ == "__main__":
    # Prompt for whether to launch with preview
    while True:
        preview_input = input("Do you want to launch the application with the preview window? (y/n): ").strip().lower()
        if preview_input in ['y', 'n']:
            use_gui = preview_input == 'y'
            break
        else:
            print("Please enter 'y' for yes or 'n' for no.")

    # Prompt for window title at the start
    window_title = input("Enter the title of the window: ").strip()
    if not window_title:
        print("Window title cannot be empty. Exiting.")
        sys.exit(1)
    print(f"Window title set to: {window_title}")

    # Replace '<<window_title>>' in all message templates
    replace_placeholder(MESSAGES_FOCUS_TEMPLATE, '<<window_title>>', window_title)
    replace_placeholder(MESSAGES_GRID_SEQUENCE_TEMPLATE, '<<window_title>>', window_title)

    # Initialize a threading Event to wait for GUI initialization
    gui_initialized_event = threading.Event()

    if use_gui:
        def start_gui():
            global gui_window
            gui_window = GUIWindow()
            gui_initialized_event.set()
            gui_window.start()

        # Start the GUI in a separate thread
        gui_thread = threading.Thread(target=start_gui, daemon=True)
        gui_thread.start()

        # Wait until the GUI is initialized
        gui_initialized_event.wait()
    else:
        gui_window = None  # Ensure gui_window is defined

    # Start the background image capture thread
    background_thread = threading.Thread(target=background_image_capture, daemon=True)
    background_thread.start()

    # Start the TCP server
    start_tcp_server()

    # Wait for initial frames to be captured
    time.sleep(10)

    # Start continuous audio recording thread
    recording_thread = threading.Thread(target=continuous_audio_recording, daemon=True)
    recording_thread.start()

    # Start transcription worker thread
    transcription_thread = threading.Thread(target=transcription_worker, daemon=True)
    transcription_thread.start()

    # Run the main asyncio event loop with ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        asyncio.run(main(executor, use_gui))
