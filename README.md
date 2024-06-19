
# AI Vision Companion

This repository features an AI vision companion/assistant that merges visual input capture with audio transcription and synthesis through various APIs and libraries. The script detects microphone input, transcribes it, processes vision input from the specified window, and produces responses using a Multimodal Large Language Model and Text-To-Speech.

## Features

- Near real-time interaction.
- Multiple monitors support.
- Captures and processes vision from a specified window.
- Transcribes audio input using Whisper.
- Synthesizes responses using text-to-speech.
- Support for GPU acceleration using CUDA.

## Installation

### Prerequisites

- Windows OS
- Python 3.10 or higher
- CUDA-compatible GPU (recommended for faster processing)
- Microphone set as the default input device in the system settings.

### Requirements
The following environment configuration was used for testing: Windows 10 Pro x64, Python 3.10.11 64-bit, and CUDA 11.8.

Install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

For CUDA 11.8(GPU):
```bash
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0 -f https://download.pytorch.org/whl/torch_stable.html  
```
### Required Environment Variables

Rename the `.env.example` file to `.env` and keep it in the root directory of the project. Fill in the missing variables with your own API keys and Eleven Labs' Voice ID - you can leave the rest unchanged:

```
OPENAI_API_KEY=your_openai_api_key
EL_API_KEY=your_eleven_labs_api_key
VOICE_ID=your_eleven_labs_voice_id

VISION_KEYWORDS=keywords-analyzing-the-sequence-of-the-last-10-seconds
FOCUS_KEYWORDS=keywords-that-focus-on-the-details-and-the-current-view

MESSAGES_TXT=default-template-gpt3.5-turbo-chat-completion-text-only-mode-keep-in-json-format

MESSAGES_FOCUS_TEMPLATE=default-template-gpt-4o-chat-completion-current-frame-only-focused-vision-mode-keep-in-json-format

MESSAGES_GRID_SEQUENCE_TEMPLATE=default-template-gpt-4o-turbo-chat-completion-sequence-vision-mode-keep-in-json-format
```

## Usage

### 1. Run the main script.
```
python visioncompanion.py
```
When running the script for the first time, it might take a while to download the `faster-whisper-large-v3` model for local use.
### 2. Type the window title.
```
Enter the title of the window: youtube
```
The script will prompt you to enter the title of the window you want to capture. 
You can specify a window by typing a simple keyword like `calculator` or `minecraft` etc. Searching process is not case-sensitive and will look for windows containing the provided keyword, even if the keyword is an incomplete word, prefix, or suffix. If you want to capture the view from your web browser's window and switch between different tabs, you can use simple keywords like `chrome` or `firefox` etc. In case you have multiple instances of an app open on multiple displays and you want to specify the tab use keywords like `youtube` or `twitch` etc. 

Only the window area specified by you will be captured on the screen. If you close the captured window/app, it will pause screen capturing. Make sure the captured window is always in the foreground and active - not minimized or in the background. (In case the window is minimized, the script will attempt to maximize it.)

### 3. Wait until the vision capture completes collecting the initial sequence of frames and speech recognition becomes active.
```
Enter the title of the window: youtube
Listening...
```

### 4. Start by speaking into your microphone :)
```
Enter the title of the window: youtube
Listening...
Registering sound...
```
### 5. Seamlessly talk about your view by naturally using vision-activating keywords during the conversation.
```
Enter the title of the window: youtube
Listening...
Registering sound...
Done.
User: It's so funny, I love what's happening in front of me right now.
```

Keywords analyzing the sequence of the last 10 seconds.
```
"see", "view", "scene", "sight", "screen", "video", "frame", "activity", "happen", "going"
```
Keywords that focus on the details and the current view.
```
"look", "focus", "attention", "recognize", "details", "carefully", "image", "picture", "place", "world", "location", "area", "action"
```
You have the option to add or remove keywords in the `.env` file.
## License

This project is licensed under the MIT License.
