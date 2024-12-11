import json, os
from utils.utils import * 

from flask import Flask, request, redirect, render_template, session, url_for, jsonify
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
# import logging
import json
import os
from dotenv import load_dotenv
from flask import Flask, request, redirect, render_template, session, url_for, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = '3d6f45a5fc12445dbac2f59c3b6c7cb1'


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
# Report Issue model
class ReportIssue(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), nullable=False)
    message = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<ReportIssue {self.name}, {self.email}>'

# Create the database tables
with app.app_context():
    db.create_all()

@app.route("/home")
def home():
    return render_template("index.html", title='SummariseIt')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('signup'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists!', 'danger')
            return redirect(url_for('signup'))
        
        # Hash the password and store user data in SQLite
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password, email=email)
        db.session.add(new_user)
        db.session.commit()
        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html', title='Sign Up')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html', title='Login')

@app.route('/set_api_key', methods=['POST'])
def set_api_key():
    data = request.get_json()
    session['api_key'] = data['apiKey']
    # logging.debug(f'API Key Saved')
    # print(data)
    return jsonify({"message": "API Key saved successfully"}), 200

videoUrl = ''
video_data = {
    'summary': '',
    'transct': dict(),
    'video_info': dict(),
    'embed_link':""
}

@app.route('/splashScreen', methods=['POST'])
def splash():
    global videoUrl, video_filepath
    videoUrl = request.form.get('videoUrl')
    videoFile = request.files.get('videoFile')
    
    # Optionally, you can save the uploaded file if necessary
    if videoFile:
        videoFile.save(f"uploads/{videoFile.filename}")  # Adjust the path as needed
        video_filepath=f"uploads/{videoFile.filename}"

    return render_template('splashScreen.html', title='SummariseIt', videoUrl=videoUrl, videoFileName=videoFile.filename if videoFile else None)

def convert_to_embed(youtube_url):
    # Check if the provided URL is a valid YouTube link
    if "youtube.com/watch?v=" in youtube_url:
        # Extract the video ID from the URL
        video_id = youtube_url.split("v=")[1].split("&")[0]  # Get the video ID
        # Return the embed link
        return f"https://www.youtube.com/embed/{video_id}"
    elif "youtu.be/" in youtube_url:
        # Handle the shortened YouTube link
        video_id = youtube_url.split("youtu.be/")[1].split("?")[0]  # Get the video ID
        return f"https://www.youtube.com/embed/{video_id}"
    else:
        return "Invalid YouTube URL"

@app.route('/process_video', methods=['POST'])
def process_video():
    global video_data
    try:
        if videoUrl:
            print('qc check here:', videoUrl)
            # videoUrl = request.args.get('videoUrl')
            # logging.debug(f'Video URL is {videoUrl}')
            
            # Attempt to retrieve the long transcript and video info
            long_transct, video_info = get_llm_transcript(videoUrl)
            # logging.debug(f'Transcript and video info generated')
            if not long_transct or not video_info:
                raise ValueError("Failed to get transcript or video information")
            
            # Process transcript and summary
            transct = get_youtube_transcript(videoUrl)
            # logging.debug(f'Process transcript for summary')
            # try:
            summary = openAI_summary(long_transct)
            print("summary",summary)
            video_data = {
            'summary': summary,
            'transct': json.dumps(transct),
            'video_info': json.dumps(video_info)
            }
            video_data['embed_link']=convert_to_embed(videoUrl)
            
        elif video_filepath:
            print(video_filepath)
            audio_path = "static/audio.mp3"
            extract_audio(video_filepath, audio_path)
            
            # Step 2: Transcribe the extracted audio
            transcription_text = transcribe_audio(audio_path)
            print(transcription_text)
            
            # Step 3: Summarize the transcription
            summary_text = summarize_transcription(transcription_text)
            print(video_filepath)
            video_data = {
            'summary': summary_text,
            'transct': json.dumps(transcription_text),
            'video_info': json.dumps(get_local_video_info(video_filepath)),
            'embed_link':video_filepath
        }
        
        
        # db.public.insert_one(video_data)
        return jsonify(success=True)
    except Exception as e:
        print(f"An error occurred in process_video: {e}")
        return jsonify(success=False, error=str(e))

import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import FreqDist
nltk.download('punkt_tab')
@app.route('/summarize')
def output():
    # Retrieve data from the database
    print('till here')
    # video_data = db.public.find_one()
    #print('summrize', video_data)
    if video_data:
        summaryv2 = video_data['summary']
        transct = json.loads(video_data['transct'])
        video_info = json.loads(video_data['video_info'])
        embed_link=video_data['embed_link']
    else:
        # Handle case where no data is found
        summaryv2 = ''
        transct = {}
        video_info = {}

    # db.public.delete_many(video_data)
    # print("check:",summaryv2[:40])
    # Prepare mind map data
    return render_template('summarize.html', title='SummariseIt', summaryv2=summaryv2,transct=transct, video_info=video_info, embed_link=embed_link)

@app.route('/process_action', methods=['POST'])
def process_action():
    action = request.json.get('action')
    summary = video_data['summary']

    transct = json.loads(video_data['transct'])
    if action == 'extract_keywords':
        response = extract_keywords_and_generate_qa(summary)
    elif action == 'generate_mcqs':
        response = generate_mcqs(summary)
    elif action == 'summarize_faqs':
        response = summarize_faqs(summary)
    elif action == 'extract_outline':
        response = extract_outline(transct)
    elif action == 'generate_notes':
        response = generate_notes(transct)
    else:
        response = {"message": "Invalid action."}

    return jsonify(response)


def extract_keywords_and_generate_qa(summary):
    # Step 1: Extract keywords
    prompt = f"Extract exactly 5 important keywords from the following text and list them as a comma-separated list: {summary}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    # Get the keywords from the response
    keywords = response.choices[0].message['content'].strip()
    # Split the keywords into a list by comma
    keywords = [keyword.strip() for keyword in keywords.split(',')]
    # Step 2: Generate questions and answers based on extracted keywords
    qa_pairs = []
    for keyword in keywords:
        question = f"What is the significance of {keyword} in the context of the given {summary}?"
        question1 = f"What is the significance of {keyword} in the context of the given text?"
        answer_prompt = f"Answer the following {question} based on this keyword: {keyword}"
        answer_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": answer_prompt}]
        )
        answer = answer_response.choices[0].message['content'].strip()
        qa_pairs.append({"question": question1, "answer": answer})
    return {
        "message": {
            "keywords": keywords,
            "qa_pairs": qa_pairs
        }
    }



def generate_mcqs(summary):
    # Step 1: Generate multiple-choice questions and options
    prompt = f"Generate 5 multiple-choice questions with 4 options each from the following text: {summary}. " \
              "Please format your response as follows: Question1: Option A, Option B, Option C, Option D. " \
              "Indicate the correct answer after each question."
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    mcq_data = response.choices[0].message['content'].strip()



    mcqs = []
    questions = mcq_data.split('\n\n')  # Split by double newline for each question block
    for question_block in questions:
        lines = question_block.strip().split('\n')
        if len(lines) >= 6:  # Ensure we have at least 5 options and a correct answer
            question = lines[0].strip()
            options = [line.strip() for line in lines[1:5]]  # Get options A-D
            correct_answer_line = lines[5].strip()
            correct_answer = correct_answer_line.split(': ')[-1].strip()  # Extract answer
            
            mcqs.append({
                "question": question,
                "options": options,
                "correct_answer": correct_answer
            })
    print({"message": mcqs})
    return {"message": mcqs}


def summarize_faqs(summary):
    prompt = f"Summarize 5 frequently asked questions based on the following text: {summary}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"message": response.choices[0].message['content'].strip()}

def extract_outline(summary):
    prompt = f"Extract an outline from the following text: {summary}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"message": response.choices[0].message['content'].strip()}

def generate_notes(summary):
    prompt = f"Generate notes from the following text: {summary}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"message": response.choices[0].message['content'].strip()}

@app.route('/report_issue', methods=['GET', 'POST'])
def report_issue():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Create a new report issue entry
        new_report = ReportIssue(name=name, email=email, message=message)
        
        try:
            db.session.add(new_report)
            db.session.commit()
            flash('Your issue has been reported!', 'success')
            return redirect(url_for('home'))  # Redirect to home or another page
        except Exception as e:
            db.session.rollback()
            flash('Error occurred while submitting the issue. Please try again.', 'error')
            print(e)
            return redirect(url_for('report_issue'))  # Stay on the report page in case of error

    return render_template('report_issue.html')  # Render the form page


@app.route('/logout')
def logout():
    # Clear the user's session
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))  # Redirect to the login page

if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=8080, debug=True)
    app.run()

