import yt_dlp
import openai
import os


openai.api_key = 'Enter api key' 

def get_youtube_video_info_and_transcribe(url):
    video_info = {}
    transcript = ""

    try:
        
        with yt_dlp.YoutubeDL() as ydl:
            info = ydl.extract_info(url, download=False)
            
            
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

    return video_info, transcript

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
            model="whisper-1",  
            file=audio_file,
            language="en"  
        )
    return response['text']

url = 'https://www.youtube.com/watch?v=wnHW6o8WMas'
video_info, transcript = get_youtube_video_info_and_transcribe(url)

print("Video Information:", video_info)
print("Transcript:", transcript)
