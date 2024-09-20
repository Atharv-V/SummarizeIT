from flask import Flask, render_template, request, jsonify
import os
import speech_recognition as sr
import moviepy.editor as mp
import concurrent.futures
import google.generativeai as genai
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
import re
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CHUNK_SIZE = 60  # Process video in 60-second chunks

try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
except ValueError as e:
    print(f"Error configuring Google Cloud API: {e}")

# api_key = os.environ.get("GOOGLE_API_KEY")  # Get API key from environment
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 512,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

# Initialize BERT-based summarization pipeline
summarizer = pipeline("summarization")

# --- Video Processing Functions ---

def convert_video_to_audio(video_path, output_audio_path):
    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path)
    clip.close()

def recognize_speech(audio_path):
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_file = r.record(source)
    try:
        result = r.recognize_google(audio_file)
    except sr.UnknownValueError:
        result = ""
    return result

def save_text_to_file(text, output_file_path):
    with open(output_file_path, mode='w', encoding='utf-8') as file:
        file.write(text)

def process_chunk(i, video_path, total_duration):
    start_time = i * CHUNK_SIZE
    end_time = min((i + 1) * CHUNK_SIZE, total_duration)
    chunk_video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'chunk_{i}.mp4')
    chunk_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f'chunk_{i}.wav')

    chunk = mp.VideoFileClip(video_path).subclip(start_time, end_time)
    chunk.write_videofile(chunk_video_path, codec='libx264', audio_codec='aac')
    convert_video_to_audio(chunk_video_path, chunk_audio_path)
    recognized_chunk_text = recognize_speech(chunk_audio_path)

    os.remove(chunk_video_path)
    os.remove(chunk_audio_path)

    return recognized_chunk_text

# Function to get YouTube transcript
def get_youtube_transcript(video_url):
    video_id_match = re.search(r"v=([a-zA-Z0-9_-]+)", video_url)
    if not video_id_match:
        raise ValueError("Invalid YouTube URL")
    video_id = video_id_match.group(1)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        raise ValueError(f"Error fetching transcript: {e}")
    result = " ".join([x['text'] for x in transcript])
    return result

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_video')
def upload_video():
    return render_template('upload_video.html')

@app.route('/youtube_link')
def youtube_link():
    return render_template('youtube_link.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video selected'}), 400

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    output_text_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'recognized_text.txt')

    clip = mp.VideoFileClip(video_path)
    total_duration = clip.duration
    clip.close()

    chunks = int(total_duration / CHUNK_SIZE) + 1
    recognized_text = ""

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_chunk = {executor.submit(process_chunk, i, video_path, total_duration): i for i in range(chunks)}
        for future in concurrent.futures.as_completed(future_to_chunk):
            recognized_chunk_text = future.result()
            recognized_text += recognized_chunk_text + "\n"

    save_text_to_file(recognized_text, output_text_file_path)

    return jsonify({'message': 'Video processed successfully! You can now summarize the text.'}), 200

@app.route('/summarize_video', methods=['POST'])
def summarize_video():
    text = request.json.get('text')
    if not text:
        return jsonify({'error': 'No text provided for summarization'}), 400

    try:
        chat_session = model.start_chat()
        response = chat_session.send_message(f"Please provide a concise summary of the following text:\n\n{text}")
        summary = response.text

        return jsonify({'summary': summary})
    except Exception as e:
        print(f"Error during summarization: {e}")
        return jsonify({'error': 'An error occurred during summarization.'}), 500

@app.route('/get_recognized_text')
def get_recognized_text():
    output_text_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'recognized_text.txt')
    try:
        with open(output_text_file_path, 'r', encoding='utf-8') as file:
            recognized_text = file.read()
        return recognized_text
    except FileNotFoundError:
        return ""  # Return an empty string if the file is not found

@app.route('/summarize_youtube', methods=['POST'])
def summarize_youtube():
    try:
        data = request.get_json()
        video_url = data.get('video_url')
        if not video_url:
            return jsonify({"error": "No video URL provided"}), 400
        
        transcript_text = get_youtube_transcript(video_url)

        summarized_text = []
        chunk_size = 1000
        total_chunks = (len(transcript_text) // chunk_size) + 1
        print("Processing... [", end="", flush=True)
        for i in range(0, len(transcript_text), chunk_size):
            chunk = transcript_text[i:i+chunk_size]
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            summarized_text.append(summary[0]['summary_text'])

            progress = (i + chunk_size) / len(transcript_text) * 100
            print("#", end="", flush=True)

        print("] 100%")

        original_text = transcript_text
        summarized_text = " ".join(summarized_text)

        response = {
            "original_text": original_text,
            "summarized_text": summarized_text
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)