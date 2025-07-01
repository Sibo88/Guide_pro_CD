import os
import wave
import logging
import numpy as np
import json
import urllib.request
import zipfile
from datetime import datetime
from flask import Flask, request, jsonify, send_file, abort
from vosk import Model, KaldiRecognizer
import firebase_admin
from firebase_admin import credentials, db
import requests
import time
from pydub import AudioSegment, effects
import language_tool_python

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Firebase setup ===
firebase_config_json = os.environ.get('FIREBASE_CONFIG')
if not firebase_config_json:
    raise Exception("Missing FIREBASE_CONFIG env var")
cred = credentials.Certificate(json.loads(firebase_config_json))
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://guidepro-9c28f-default-rtdb.firebaseio.com/'
})

# === AssemblyAI setup ===
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")

# === Paths & model info ===
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "models")
model_path = os.path.join(model_dir, "vosk-model-small-en-in-0.4")
model_zip_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip"
model_zip_path = os.path.join(base_dir, "model.zip")

raw_audio_path = os.path.join(base_dir, "recorded_audio.raw")
wav_audio_path = os.path.join(base_dir, "audio_file.wav")
transcription_path = os.path.join(base_dir, "transcription.txt")
feedback_path = os.path.join(base_dir, "feedback_file.txt")
summary_path = os.path.join(base_dir, "summary.txt")

# Ensure text files exist
for path in [transcription_path, feedback_path, summary_path]:
    open(path, 'a').close()

# === Download Vosk model if missing ===
def download_model():
    if not os.path.exists(model_path):
        logging.info("Downloading Vosk model...")
        os.makedirs(model_dir, exist_ok=True)
        urllib.request.urlretrieve(model_zip_url, model_zip_path)
        with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        os.remove(model_zip_path)
    else:
        logging.info("Vosk model already exists.")

# === Preprocess audio (mono, normalize, 16kHz) ===
def preprocess_audio():
    logging.info("Preprocessing audio...")
    sound = AudioSegment.from_raw(raw_audio_path, sample_width=2, frame_rate=48000, channels=1)
    sound = sound.set_channels(1).set_frame_rate(16000)
    sound = effects.normalize(sound)
    sound.export(wav_audio_path, format="wav")
    logging.info("Audio preprocessed and saved as WAV.")

# === Transcribe with Vosk ===
def transcribe_with_vosk():
    download_model()
    model = Model(model_path)
    rec = KaldiRecognizer(model, 16000)
    with wave.open(wav_audio_path, 'rb') as wf:
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            rec.AcceptWaveform(data)
    result = json.loads(rec.FinalResult())
    text = result.get("text", "")
    logging.info(f"Vosk text: '{text}'")
    return text

# === Transcribe with AssemblyAI (advanced) ===
def transcribe_with_assemblyai():
    logging.info("Uploading to AssemblyAI...")
    with open(wav_audio_path, 'rb') as f:
        upload_resp = requests.post(
            'https://api.assemblyai.com/v2/upload',
            headers={'authorization': ASSEMBLYAI_API_KEY},
            files={'file': f}
        )
    upload_url = upload_resp.json()['upload_url']

    logging.info("Starting transcription...")
    trans_resp = requests.post(
        'https://api.assemblyai.com/v2/transcript',
        json={
            'audio_url': upload_url,
            'punctuate': True,
            'format_text': True,
            'disfluencies': True
        },
        headers={'authorization': ASSEMBLYAI_API_KEY}
    )
    tid = trans_resp.json()['id']

    while True:
        poll = requests.get(f'https://api.assemblyai.com/v2/transcript/{tid}',
                            headers={'authorization': ASSEMBLYAI_API_KEY})
        status = poll.json()['status']
        if status == 'completed':
            text = poll.json()['text']
            logging.info(f"AssemblyAI text: '{text}'")
            return text
        elif status == 'failed':
            logging.error("AssemblyAI failed.")
            return ""
        time.sleep(3)

# === Grammar correction ===
def grammar_correct(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    corrected = language_tool_python.utils.correct(text, matches)
    logging.info(f"Corrected text: '{corrected}'")
    return corrected

# === Analyze & push to Firebase ===
def analyze_and_push(text):
    with open(transcription_path, 'w') as f:
        f.write(text)

    words = text.split()
    total = len(words)
    word_count = {}
    for w in words:
        w = w.lower().strip(".,!?;:\"'()[]{}")
        word_count[w] = word_count.get(w, 0) + 1
    repetitive = {w: c for w, c in word_count.items() if c > 1}
    filler = {w: word_count.get(w, 0) for w in {'uh', 'um', 'so', 'ah', 'because'}}
    
    save_feedback(repetitive, filler, total)
    save_summary(total, filler, repetitive)

    pres_score = max(100 - len(repetitive)*2, 0)
    time_score = max(100 - sum(filler.values())*2, 0)
    overall = (pres_score+time_score)/2

    now = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    user_id = "user_1"
    ref = db.reference(f'guidpro_results/{user_id}/session_{now}')
    ref.set({
        'transcription': text,
        'feedback': {'repetitive_words': repetitive, 'filler_words': filler, 'total_word_count': total},
        'summary': {'presentation_score': pres_score, 'time_score': time_score, 'overall_score': overall}
    })
    logging.info(f"Pushed to Firebase under {user_id}/session_{now}")

def save_feedback(repetitive, filler, total):
    with open(feedback_path, 'w') as f:
        f.write("=== Feedback ===\nRepetitive Words:\n")
        for k,v in repetitive.items(): f.write(f"{k}: {v}\n")
        f.write("\nFiller Words:\n")
        for k,v in filler.items(): f.write(f"{k}: {v}\n")
        f.write(f"\nTotal Word Count: {total}\n")

def save_summary(total, filler, repetitive):
    pres = max(100 - len(repetitive)*2,0)
    time = max(100 - sum(filler.values())*2,0)
    ov = (pres+time)/2
    with open(summary_path,'w') as f:
        f.write(f"=== Summary ===\nPresentation Score: {pres}\nTime Score: {time}\nOverall Score: {ov}\n")

# === Upload route with smart fallback ===
@app.route('/upload', methods=['POST'])
def upload_audio():
    try:
        if 'file' not in request.files: return jsonify({"error":"No file"}),400
        f = request.files['file']
        f.save(raw_audio_path)
        preprocess_audio()

        vosk_text = transcribe_with_vosk()
        filler_count = sum(1 for w in vosk_text.split() if w in {'uh','um','so','ah','because'})

        # fallback if short OR many fillers
        if len(vosk_text.split())<5 or filler_count>5:
            logging.info("Fallback to AssemblyAI...")
            text = transcribe_with_assemblyai()
        else:
            text = vosk_text

        corrected = grammar_correct(text)
        analyze_and_push(corrected)
        return jsonify({"message":"Processed & uploaded!", "transcript":corrected})
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error":str(e)}),500

# === Other routes unchanged ===
@app.route('/')
def home(): return "✅ GuidPro improved server running!"

@app.route('/transcription.txt')
def serve_transcription(): return serve_file(transcription_path,"Not found.")

@app.route('/feedback.txt')
def serve_feedback(): return serve_file(feedback_path,"Not found.")

@app.route('/summary.txt')
def serve_summary(): return serve_file(summary_path,"Not found.")

@app.route('/audio.wav')
def serve_audio(): return serve_file(wav_audio_path,"Not found.")

def serve_file(path,msg):
    return send_file(path) if os.path.exists(path) else abort(404,description=msg)

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
