from collections import defaultdict
from langchain_community.document_loaders import YoutubeLoader
#from openai import OpenAI
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.summarize import load_summarize_chain
# from langchain.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import openai
openai.api_key = ''

# promat template
prompt_template1 = """## Video Content Distillation

As a specialist in distilling video content, your task is to extract and present the essence of a YouTube video transcript in two formats:

1. **Concise Summary**: 
    Craft a succinct, engaging summary that encapsulates the video's major points, key insights, and overarching conclusions. Your summary should provide a comprehensive snapshot, allowing readers to grasp the content's significance quickly.

2. **Bullet-Pointed Highlights**: 
    Create a series of 6-7 bullet points, each highlighting a key insight, moment, or takeaway from the video. Incorporate emojis for added visual interest and ensure each point is clear and impactful.

**Output Format:** Use Markdown to format your work. Begin with the concise summary, followed by the bullet-pointed highlights. This structured approach will ensure clarity and ease of reading."""


import yt_dlp
import openai
import os



def get_llm_transcript(url):
    video_info = {}
    transcript = ""

    try:
        # Extract video information
        with yt_dlp.YoutubeDL() as ydl:
            info = ydl.extract_info(url, download=False)
            
            # Extract necessary video information
            video_info = {
                "Title": info.get("title"),
                "Description": info.get("description"),
                "Views": info.get("view_count"),
                "Length (seconds)": info.get("duration"),
                "Rating": info.get("average_rating"),
                "Author": info.get("uploader"),
                "Published Date": info.get("upload_date"),
                "Thumbnail URL": info.get("thumbnail")
            }

        # Download the audio of the video
        download_audio(url)

        # Transcribe the audio file
        transcript = transcribe_audio('audio.mp3')

    except Exception as e:
        return f"Error: {e}"

    # Optionally, clean up the downloaded audio file
    os.remove('audio.mp3')

    return  transcript,video_info

def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def transcribe_audio(file_path):
    with open(file_path, 'rb') as audio_file:
        response = openai.Audio.transcribe(
            model="whisper-1",  # Specify the Whisper model
            file=audio_file,
            language="en"  # Specify the language if needed
        )
    return response['text']

# # Example usage
# url = 'https://www.youtube.com/watch?v=wnHW6o8WMas'
# video_info, transcript = get_youtube_video_info_and_transcribe(url)

# print("Video Information:", video_info)
# print("Transcript:", transcript)


def get_youtube_transcript(youtube_url):
    video_id = youtube_url.split("youtube.com/watch?v=")[-1]
    print(video_id)
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    # Attempt to directly select the preferred or fallback language
    try:
        transcript = transcript_list.find_transcript(['en-GB', 'en'])
    except Exception as e:  # Catch more specific exceptions if possible
        print(f"Error finding transcript: {e}")
        return {}
    
    print('here1')
    captions_dict = defaultdict(str)  # Use string directly to concatenate

    for caption in transcript.fetch():
        start_time = round(caption['start'] / 60) * 60
        caption_text = caption['text'].strip()
        # Append directly to the string for each key, reducing list overhead
        captions_dict[start_time] += caption_text + "\n"
    
    captions_final = {}
    for time, captions in captions_dict.items():
        minutes, seconds = divmod(time, 60)
        time_formatted = f"{minutes:02d}:{seconds:02d}"
        # Trim the trailing newline when finalizing the string
        captions_final[time_formatted] = captions[:-1]  # Remove the last newline
    
    return captions_final

def openAI_summary(input_text):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0125",
    messages=[
        {
        "role": "system",
        "content": prompt_template1
        },
        {
        "role": "user",
        "content": f"**Transcript:** '''{input_text}'''"
        },
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response.choices[0].message.content


# def openAI_summary(transct_text, api_key, type = 'summary'):
#     # print('open api', api_key)
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=9000, chunk_overlap=100)
#     llm = ChatOpenAI(temperature=0, model ='gpt-3.5-turbo-0125', openai_api_key=api_key, max_tokens=300, streaming=False)
#     texts = text_splitter.split_documents(transct_text)
#     # print('here4')
#     prompt_template = prompt_template1
#     try:
#         PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
#         # print(PROMPT)
#         chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False, map_prompt=PROMPT, combine_prompt=PROMPT)
#         print('here45')
#         # print(chain)
#         response = chain.invoke(texts)
#         print('here5')
#         return response
#     except:
#         return 'Some error in API'

from moviepy.editor import VideoFileClip
import openai

# Set your OpenAI API key

# Step 1: Extract Audio from Video
def extract_audio(video_path, output_audio_path):
    # Load the video clip
    video = VideoFileClip(video_path)
    
    # Extract the audio from the video
    audio = video.audio
    
    # Save the extracted audio as an MP3 file
    audio.write_audiofile(output_audio_path)
    
    # Close video and audio to release resources
    audio.close()
    video.close()
    
    print(f"Audio extracted to {output_audio_path}")




def convert_seconds_to_mins_secs(seconds):
    """Convert seconds to a string in min:secs format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"  # Format to ensure two digits for seconds

def transcribe_audio(audio_path):
    # Open the extracted audio file
    with open(audio_path, "rb") as audio_file:
        # Transcribe the audio using Whisper
        transcription_response = openai.Audio.transcribe("whisper-1", audio_file, response_format="verbose_json")
    
    # Initialize an empty dictionary to hold timestamps and sentences
    transcription_dict = {}
    
    # Process the transcription response
    for segment in transcription_response['segments']:
        start_time = segment['start']  # Start time of the segment
        text = segment['text']          # Transcribed text
        formatted_time = convert_seconds_to_mins_secs(start_time)  # Convert to min:secs format
        transcription_dict[formatted_time] = text.strip()  # Add to the dictionary

    # Save the full transcription to a text file (if needed)
    with open('transcription.txt', 'w') as f:
        f.write(transcription_response['text'])

    print("Transcription completed!")
    return transcription_dict


from moviepy.editor import VideoFileClip

def get_local_video_info(video_path):
    """Fetch video information from a local video file."""
    # Load the video file
    clip = VideoFileClip(video_path)
    
    # Gather information
    video_info = {
        "Title": video_path.split('/')[-1],  # Use filename as title
        "duration": clip.duration,            # Duration in seconds
        "fps": clip.fps,                      # Frames per second
        "resolution": clip.size,              # Resolution (width, height)
    }

    # Close the video clip to free resources
    clip.close()

    return video_info





import google.generativeai as genai
# Step 3: Summarize the Transcription using GPT-4
def summarize_transcription(transcription_text):
    # Summarize the transcription using GPT-3.5 Turbo
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # GPT-3.5 Turbo model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the following text:\n\n{transcription_text}"}
        ],
        max_tokens=150,  # Adjust token length for summary size
        temperature=0.5
    )
    
    # Get the summary from the chat response
    summary_text = response['choices'][0]['message']['content']
    
    # Save the summary to a text file
    with open('summary.txt', 'w') as f:
        f.write(summary_text)
    
    print("Summary completed!")
    return summary_text
        


