
# AI Vision Companion (SteamVR Overlay Prototype)

This repository features an AI vision companion/assistant that merges visual input capture with audio transcription and synthesis through various APIs and libraries. The script detects microphone input, transcribes it, processes vision input from the specified window, creates very detailed caption with Florence-2, and produces responses using a Large Language Model (OpenAI API) and Default Windows TTS.

## Features

- Near real-time interaction.
- Multiple monitor support.
- Captures and processes vision locally from a specified window.
- Transcribes audio input locally using Whisper-Large-3-Turbo model.
- Synthesizes responses locally using Windows default text-to-speech.
- Support for GPU acceleration using CUDA.

## Installation

### Prerequisites

- Windows OS
- Python 3.10 or higher
- CUDA-compatible GPU
- Microphone set as the default input device in the system settings.

### Requirements
The following environment configuration was used for testing: Windows 10 Pro x64, Python 3.10.11 64-bit, and CUDA 11.8.

Install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

Install torch with your CUDA version, e.g. :
```bash
pip install torch==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torchvision==0.18.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### Required Environment Variables

Rename the `.env.example` file to `.env` and keep it in the root directory of the project. Fill in the missing `OPENAI_API_KEY=xxxxxxxx...` variable with your own API key.

```
OPENAI_API_KEY=your_openai_api_key
VISION_KEYWORDS=scene,sight,video,frame,activity,happen,going
```

## Usage

### 1. Launch SteamVR. 

### 2. Run the main script.
```
python visioncompanion.py
```
When running the script for the first time, it might take a while to download the speech recognition model `faster-whisper-large-v3` as well as image captioning model `florence-2-base-ft` for local use.

### 3. Choose if you want to launch the app with vision capture preview window (`y` for yes or `n` for no).
```
Using cuda:0 device
Loading Florence-2 model...
Florence-2 model loaded.
Do you want to launch the application with the preview window? (y/n):
```
### 4. Type the window title.
```
Do you want to launch the application with the preview window? (y/n): y
Enter the title of the window:
```
The script will prompt you to enter the title of the window you want to capture. 
You can specify a window by typing a simple keyword like `calculator` or `minecraft` etc. Searching process is not case-sensitive and will look for windows containing the provided keyword, even if the keyword is an incomplete word, prefix, or suffix. If you want to capture the view from your web browser's window and switch between different tabs, you can use simple keywords like `chrome` or `firefox` etc. In case you have multiple instances of an app open on multiple displays and you want to specify the tab use keywords like `youtube` or `twitch` etc. 

Make sure the captured window is always in the foreground and active - not minimized or in the background. (In case the window is minimized, the script will attempt to maximize it.)

### 5. Wait until the vision capture completes collecting the initial sequence of frames and speech recognition becomes active.
```
Enter the title of the window: youtube
Window title set to: youtube
Starting continuous audio recording...
```

### 6. Start by speaking into your microphone :)
```
Starting continuous audio recording...
User: Hi there, how are you doing?
```
### 7. Download and unzip `VRChatCompanion.zip`. Launch `VRCompanion.exe`. Minimize the window so it doesn't cover the window of the application you want to capture in real-time. 

### 8. Seamlessly talk about your view by naturally using vision-activating keywords during the conversation.
```
User: Hi there, how are you doing?
Assistant: Just hanging out, enjoying the vibes.
User:Can you describe what you can see on the screen?
Assistant: There's an anime girl with long blonde hair, looking stylish in a red dress.
User: How can I say it in Spanish?
Assistant: "Chica con cabello rubio y vestido rojo." Simple and stylish!
```

Keywords analyzing the sequence of the last 10 seconds.
```
"scene", "sight",  "video", "frame", "activity", "happen", "going"
```
You have the option to add or remove keywords in the `.env` file.

If you receive response: `Assistant: Error in generating response` make sure to update OpenAI API key inside `.env`. Your API key might be incorrect or missing - this variable cannot be empty.

## License

This project is open-source under the MIT license—no need to credit me. The only thing I’d like to ask is, if you’re reading this and planning on reusing any part of the code from this repository in your own projects, to "consider" sharing it open-source later. It's not required, but that would make me happy if we can collectively push things one step forward for everyone else :)
