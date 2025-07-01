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

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Firebase setup ===
firebase_config_json = os.environ.get('FIREBASE_CONFIG')
if not firebase_config_json:
    raise Exception("Missing FIREBASE_CONFIG environment variable")
cred = credentials.Certificate(json.loads(firebase_config_json))
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://guidepro-9c28f-default-rtdb.firebaseio.com/'
})

# === AssemblyAI setup ===
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")  # store your API key as env var

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
    if not os.path.exists(path):
        open(path, 'w').close()

# === Download Vosk model if missing ===
def download_model():
    if not os.path.exists(model_path):
        logging.info("Vosk model not found. Downloading...")
        os.makedirs(model_dir, exist_ok=True)
        urllib.request.urlretrieve(model_zip_url, model_zip_path)
        logging.info("Extracting model...")
        with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        os.remove(model_zip_path)
        logging.info("Model ready.")
    else:
        logging.info("Vosk model already exists.")

# === Convert raw audio to WAV ===
def convert_to_wav():
    try:
        logging.info("Converting raw audio to WAV...")
        with open(raw_audio_path, 'rb') as raw_file:
            raw_data = raw_file.read()
        audio_data = np.frombuffer(raw_data, dtype=np.int16)
        with wave.open(wav_audio_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(48000)
            wav_file.writeframes(audio_data.tobytes())
        logging.info("WAV created.")
    except Exception as e:
        logging.error(f"Error in WAV conversion: {e}")

# === Transcribe using Vosk ===
def transcribe_with_vosk():
    download_model()
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 48000)
    text = ""
    with wave.open(wav_audio_path, 'rb') as wf:
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            recognizer.AcceptWaveform(data)
    result = json.loads(recognizer.FinalResult())
    text = result.get("text", "")
    logging.info(f"Vosk transcription: '{text}'")
    return text

# === Transcribe using AssemblyAI ===
def transcribe_with_assemblyai():
    logging.info("Uploading to AssemblyAI...")
    with open(wav_audio_path, 'rb') as f:
        upload_response = requests.post(
            'https://api.assemblyai.com/v2/upload',
            headers={'authorization': ASSEMBLYAI_API_KEY},
            files={'file': f}
        )
    upload_url = upload_response.json()['upload_url']

    logging.info("Starting transcription at AssemblyAI...")
    transcript_response = requests.post(
        'https://api.assemblyai.com/v2/transcript',
        json={'audio_url': upload_url},
        headers={'authorization': ASSEMBLYAI_API_KEY}
    )
    transcript_id = transcript_response.json()['id']

    # Poll for completion
    while True:
        polling_response = requests.get(
            f'https://api.assemblyai.com/v2/transcript/{transcript_id}',
            headers={'authorization': ASSEMBLYAI_API_KEY}
        )
        status = polling_response.json()['status']
        if status == 'completed':
            text = polling_response.json()['text']
            logging.info(f"AssemblyAI transcription: '{text}'")
            return text
        elif status == 'failed':
            logging.error("AssemblyAI transcription failed.")
            return ""
        else:
            time.sleep(3)

# === Analyze & push results ===
def analyze_and_push(text):
    # Save transcription
    with open(transcription_path, 'w') as f:
        f.write(text)

    words = text.split()
    word_count = {}
    for word in words:
        word = word.lower().strip(".,!?;:\"'()[]{}")
        word_count[word] = word_count.get(word, 0) + 1
    repetitive = {w: c for w, c in word_count.items() if c > 1}
    filler = {w: word_count.get(w, 0) for w in {"uh", "ah", "um", "so", "because"}}
    total = len(words)

    save_feedback(repetitive, filler, total)
    save_summary(total, filler, repetitive)

    now = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    session_id = f"session_{now}"
    user_id = "user_1"  # replace with real user ID

    ref = db.reference(f'guidpro_results/{user_id}/{session_id}')
    pres_score = max(100 - len(repetitive) * 2, 0)
    time_score = max(100 - sum(filler.values()) * 2, 0)
    overall = (pres_score + time_score) / 2

    ref.set({
        'transcription': text,
        'feedback': {
            'repetitive_words': repetitive,
            'filler_words': filler,
            'total_word_count': total
        },
        'summary': {
            'presentation_score': pres_score,
            'time_score': time_score,
            'overall_score': overall
        }
    })
    logging.info(f"Pushed results to Firebase under {user_id}/{session_id}")

def save_feedback(repetitive, filler, total):
    try:
        with open(feedback_path, 'w') as f:
            f.write("=== Feedback ===\n\nRepetitive Words:\n")
            for w, c in repetitive.items():
                f.write(f"{w}: {c}\n")
            f.write("\nFiller Words:\n")
            for w, c in filler.items():
                f.write(f"{w}: {c}\n")
            f.write(f"\nTotal Word Count: {total}\n")
        logging.info("Feedback saved locally.")
    except Exception as e:
        logging.error(f"Error saving feedback: {e}")

def save_summary(total, filler, repetitive):
    try:
        pres_score = max(100 - len(repetitive) * 2, 0)
        time_score = max(100 - sum(filler.values()) * 2, 0)
        overall = (pres_score + time_score) / 2
        with open(summary_path, 'w') as f:
            f.write("=== Summary ===\n")
            f.write(f"Presentation Score: {pres_score}\n")
            f.write(f"Time Score: {time_score}\n")
            f.write(f"Overall Score: {overall}\n")
        logging.info("Summary saved locally.")
    except Exception as e:
        logging.error(f"Error saving summary: {e}")

# === Routes ===
@app.route('/')
def home():
    return "✅ GuidPro server running! POST raw audio to /upload"

@app.route('/upload', methods=['POST'])
def upload_audio():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in request"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        file.save(raw_audio_path)
        logging.info("Audio uploaded. Starting processing...")
        convert_to_wav()

        # Hybrid transcription: Vosk first, fallback to AssemblyAI
        text = transcribe_with_vosk()
        if len(text.split()) < 5:
            logging.info("Vosk result too short → fallback to AssemblyAI...")
            text = transcribe_with_assemblyai()

        analyze_and_push(text)
        return jsonify({"message": "Processed & uploaded to Firebase!", "transcript": text})
    except Exception as e:
        logging.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/transcription.txt')
def serve_transcription():
    return serve_file(transcription_path, "Transcription not found.")

@app.route('/feedback.txt')
def serve_feedback():
    return serve_file(feedback_path, "Feedback not found.")

@app.route('/summary.txt')
def serve_summary():
    return serve_file(summary_path, "Summary not found.")

@app.route('/audio.wav')
def serve_audio():
    return serve_file(wav_audio_path, "Audio not found.")

def serve_file(path, not_found_msg):
    if os.path.exists(path):
        return send_file(path, as_attachment=False)
    else:
        abort(404, description=not_found_msg)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
