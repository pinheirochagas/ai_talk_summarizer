#%%
import moviepy.editor as mp
import openai
import requests
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment

import os
# Get the absolute path of the script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))
# Change the working directory to the script's directory
os.chdir(script_directory)
# Print the current working directory to verify the change
print(f"Current working directory: {os.getcwd()}")

#%% Set API keys
whisper_api_key = input("Please enter your Whisper API key: ")
gpt3_api_key = input("Please enter your GPT API key: ")
openai.api_key = whisper_api_key

#%%
def load_video_and_extract_audio(video_path, audio_path):
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)

def trimm_audio(audio_path, time):
    audio_cut = AudioSegment.from_mp3(audio_path)
    # PyDub handles time in milliseconds
    #time = 10 * 60 * 1000
    audio_cut = audio_cut[:time]
    audio_cut.export(audio_path, format="mp3")

def transcribe_audio_whisper_local(audio_path):
    # Run the 'whisper' command and capture its output
    audio_file = open(audio_path, "rb")
    transcription = openai.Audio.transcribe("whisper-1", audio_file)
    return transcription

def summarize_text_gpt3(text, gpt3_api_key, summary_file_path):
    # Send the text to the GPT-3 API for summarization
    url = "https://api.openai.com/v1/engines/davinci/completions"
    headers = {
        "Authorization": f"Bearer {gpt3_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": f"Please summarize the following text:\n{text}\n",
        "max_tokens": 100  # Limit the response to 100 tokens
    }
    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    # Check the status code of the API response
    if response.status_code == 200:
        # Successful API call
        summary = result['choices'][0].get('text', '').strip()

        # Save the summary to a text file
        with open(summary_file_path, "w") as file:
            file.write(summary)
        print(f"Summary saved to: {summary_file_path}")

        return summary
    else:
        # Unsuccessful API call
        print(f"GPT-3 API call was unsuccessful. Status code: {response.status_code}")
        print("Error message (if any):", result.get('error', 'No error message available'))
        return None

#%%
# Set your video file path, audio file path, Whisper model and processor paths, and GPT-3 API key
video_path = "/Users/pedro_SRCD_final.mp4" 
# get the video here: https://drive.google.com/file/d/1ymxx7JdSoPseCDrXZ1ErsVjaJEXDOoEh/view
audio_path = script_directory + "/talks/pedro_SRCD_final.mp3"

#%%
# Step 1: Load the video and extract the audio
load_video_and_extract_audio(video_path, audio_path)

#%%
# Step 1.5 cut audio
trimm_audio(audio_path, 600000) # maybe iterate this to save different chunks?

#%%
# Step 2: Transcribe the audio using the locally installed Whisper ASR model
transcription = transcribe_audio_whisper_local(audio_path)

#%%
# Step 3: Summarize the transcribed text using GPT-3

summary_file_path = "/Users/pinheirochagas/Pedro/Stanford/code/ai_talk_summarizer/talks/pedro_SRCD_final.txt"
summary = summarize_text_gpt3(transcription, gpt3_api_key, summary_file_path)
# %%
