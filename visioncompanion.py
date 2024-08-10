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
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import soundfile as sf
import psutil

# Set up logging
# logging.basicConfig(filename='app.log', level=logging.DEBUG, 
#                     format='%(asctime)s %(levelname)s:%(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
OPENAI_API_KEY = openai.api_key
XTTS2_PATH = os.getenv('XTTS2_PATH')
XTTS2_CONFIG = os.getenv('XTTS2_CONFIG')
XTTS2_SAMPLE = os.getenv('XTTS2_SAMPLE')

sys.stdout.reconfigure(encoding='utf-8')

model_size = "large-v3"
whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Load XTTS configuration
xtts_config = XttsConfig()
xtts_config.load_json(XTTS2_CONFIG)

# Initialize XTTS model
xtts_model = Xtts.init_from_config(xtts_config)
xtts_model.load_checkpoint(xtts_config, checkpoint_dir=XTTS2_PATH, eval=True)
xtts_model.cuda()  # Move the model to GPU if available

audio_queue = queue.Queue()
playback_thread = None
playback_stop_event = threading.Event()
stop_recording_event = threading.Event()

def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    logging.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")

def audio_playback_worker():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    output=True)
    
    while not playback_stop_event.is_set():
        try:
            audio_segment = audio_queue.get(timeout=0.1)
            chunk_size = 512
            for i in range(0, len(audio_segment), chunk_size):
                if playback_stop_event.is_set():
                    break
                chunk = audio_segment[i:i+chunk_size].raw_data
                stream.write(chunk)
            audio_queue.task_done()
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

async def stream_xtts2(text, sample_path):
    try:
        log_memory_usage()
        outputs = xtts_model.synthesize(
            text,
            xtts_config,
            speaker_wav=sample_path,
            gpt_cond_len=10,
            temperature=0.9,
            top_p=0.7,
            length_penalty=1.0,
            repetition_penalty=6.0,
            language='en',
            speed=1.35,
            enable_text_splitting=False
        )
        
        audio = outputs['wav']
        sample_rate = xtts_config.audio.sample_rate
        
        # Convert the audio to a bytes object
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='wav')
        buffer.seek(0)
        
        return buffer.getvalue()
    except Exception as e:
        logging.error(f"Error in stream_xtts2: {e}")
        return None

async def process_and_play_streaming(audio_data):
    try:
        if audio_data:
            audio_segment = AudioSegment.from_wav(io.BytesIO(audio_data))
            audio_segment = audio_segment.set_channels(1).set_frame_rate(44100)
            audio_queue.put(audio_segment)
            start_audio_playback_thread()
    except Exception as e:
        logging.error(f"Error in process_and_play_streaming: {e}")

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
    "Goodbye.","Thanks for watching!", "Thank you for watching!", "I feel like I'm going to die.", "Thank you for watching.", "Transcription by CastingWords"
]

async def transcribe_with_whisper(audio_file):
    try:
        segments, info = whisper_model.transcribe(audio_file, beam_size=5, language="en")
        transcription = ""
        for segment in segments:
            transcription += segment.text + " "
        transcription = transcription.strip()

        if transcription in whisper_hallucinated_phrases:
            transcription = "Please continue."
        
        return transcription
    except Exception as e:
        logging.error(f"Error in transcribe_with_whisper: {e}")
        return "Error in transcription"

def detect_microphone_input(threshold, check_duration=17):
    try:
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
        
        return "Share what you think is happening."
    except Exception as e:
        logging.error(f"Error in detect_microphone_input: {e}")
        return False

def record_audio_with_threshold(file_path, threshold, max_silence_duration=1):
    try:
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
    except Exception as e:
        logging.error(f"Error in record_audio_with_threshold: {e}")

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

async def main():
    global stop_recording_event
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

    while True:
        try:
            threshold = 250
            stop_recording_event.clear()
            mic_input_result = detect_microphone_input(threshold)
            
            if mic_input_result == True:
                stop_audio_playback()  # Stop any ongoing playback
                audio_file = "temp_recording.wav"
                record_audio_with_threshold(audio_file, threshold)
                stop_recording_event.set()  # Interrupt any ongoing recording
                input_text = await transcribe_with_whisper(audio_file)
                print(f"User: {input_text}")
                os.remove(audio_file)
            elif mic_input_result == "Share what you think is happening.":
                input_text = mic_input_result
                print(f"User: {input_text}")
            else:
                continue  # Skip this iteration if False is returned

            if any(word in input_text.lower() for word in focus_keywords):
                vision_feed_grid_resized, vision_feed_current_frame_resized = await capture_vision_input()
                if vision_feed_grid_resized is not None and vision_feed_current_frame_resized is not None:
                    image_frame = image_to_base64_data_uri(vision_feed_current_frame_resized)
                    messages_focus = [message.copy() for message in messages_focus_template]
                    messages_focus[1]['content'][0]['image_url']['url'] = image_frame

                model = "gpt-4o-2024-08-06"
                messages = messages_focus
                max_tokens = 256
                logging.info("gpt-4o-focus-mode")

            elif any(word in input_text.lower() for word in vision_keywords):
                vision_feed_grid_resized, vision_feed_current_frame_resized = await capture_vision_input()
                if vision_feed_grid_resized is not None and vision_feed_current_frame_resized is not None:
                    image_grid = image_to_base64_data_uri(vision_feed_grid_resized)
                    messages_grid_sequence = [message.copy() for message in messages_grid_sequence_template]
                    messages_grid_sequence[1]['content'][0]['image_url']['url'] = image_grid
                    
                model = "gpt-4o-2024-08-06"
                messages = messages_grid_sequence
                max_tokens = 256
                logging.info("gpt-4o-grid-sequence-mode")

            else:
                model = "gpt-4o-mini"
                messages = messages_txt
                max_tokens = 150
                logging.info("gpt-4o-mini-text-mode")

            conversation_history.append({"role": "user", "content": input_text})

            full_response = ""
            sentence_buffer = ""
            async for sentence in stream_openai_response(client, model, messages + conversation_history, max_tokens, 0.35, 0.9, 1.2, 1.1):
                full_response += sentence + " "
                sentence_buffer += sentence + " "
                
                # Process and queue audio for complete sentences
                if sentence.strip().endswith(('.', '!', '?')):
                    print(f"Assistant: {sentence_buffer.strip()}")
                    audio_data = await stream_xtts2(sentence_buffer.strip(), XTTS2_SAMPLE)
                    if audio_data:
                        await process_and_play_streaming(audio_data)
                    else:
                        logging.warning("Failed to generate audio, skipping playback")
                    sentence_buffer = ""

            # Process any remaining text in the buffer
            if sentence_buffer.strip():
                print(f"Assistant: {sentence_buffer.strip()}")
                audio_data = await stream_xtts2(sentence_buffer.strip(), XTTS2_SAMPLE)
                if audio_data:
                    await process_and_play_streaming(audio_data)
                else:
                    logging.warning("Failed to generate audio, skipping playback")

            conversation_history.append({"role": "assistant", "content": full_response.strip()})

        except Exception as e:
            logging.error(f"An error occurred in main loop: {e}")
            continue

        finally:
            # Clean up resources
            torch.cuda.empty_cache()
            log_memory_usage()

# Start the background image capture thread
background_thread = threading.Thread(target=background_image_capture, daemon=True)
background_thread.start()

# Wait for initial frames to be captured
time.sleep(10)

# Run the main asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())
