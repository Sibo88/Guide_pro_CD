import os
import wave
import logging
import numpy as np
import json
import urllib.request
import zipfile
from flask import Flask, request, jsonify, send_file, abort
from vosk import Model, KaldiRecognizer

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "models")
model_path = os.path.join(model_dir, "vosk-model-small-en-us-0.15")
model_zip_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
model_zip_path = os.path.join(base_dir, "model.zip")

raw_audio_path = os.path.join(base_dir, "recorded_audio.raw")
wav_audio_path = os.path.join(base_dir, "audio_file.wav")
transcription_path = os.path.join(base_dir, "transcription.txt")
feedback_path = os.path.join(base_dir, "feedback_file.txt")
summary_path = os.path.join(base_dir, "summary.txt")

# Ensure output files exist
for path in [transcription_path, feedback_path, summary_path]:
    if not os.path.exists(path):
        open(path, 'w').close()

def download_model():
    if not os.path.exists(model_path):
        logging.info("Vosk model not found. Downloading...")
        os.makedirs(model_dir, exist_ok=True)
        urllib.request.urlretrieve(model_zip_url, model_zip_path)
        logging.info("Model zip downloaded. Extracting...")
        with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        os.remove(model_zip_path)
        logging.info("Model extracted and zip deleted.")
    else:
        logging.info("Vosk model already exists.")

def convert_to_wav():
    try:
        logging.info("Converting raw audio to WAV...")
        with open(raw_audio_path, 'rb') as raw_file:
            raw_data = raw_file.read()
        audio_data = np.frombuffer(raw_data, dtype=np.int16)
        with wave.open(wav_audio_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(44100)
            wav_file.writeframes(audio_data.tobytes())
        logging.info("WAV file created.")
    except Exception as e:
        logging.error(f"Error during WAV conversion: {e}")

def transcribe_audio():
    try:
        logging.info("Transcribing audio...")
        download_model()
        model = Model(model_path)
        recognizer = KaldiRecognizer(model, 44100)
        with wave.open(wav_audio_path, 'rb') as wf:
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                recognizer.AcceptWaveform(data)
        result = json.loads(recognizer.FinalResult())
        text = result.get("text", "")
        with open(transcription_path, 'w') as f:
            f.write(text)
        logging.info("Transcription complete.")
        analyze_text(text)
    except Exception as e:
        logging.error(f"Error during transcription: {e}")

def analyze_text(text):
    try:
        words = text.split()
        word_count = {}
        for word in words:
            word = word.lower().strip(".,!?;:\"'()[]{}")
            word_count[word] = word_count.get(word, 0) + 1
        repetitive = {word: count for word, count in word_count.items() if count > 1}
        filler = {word: word_count.get(word, 0) for word in {"uh", "ah", "um", "so", "because"}}
        total = len(words)
        save_feedback(repetitive, filler, total)
        save_summary(total, filler, repetitive)
    except Exception as e:
        logging.error(f"Error analyzing text: {e}")

def save_feedback(repetitive, filler, total):
    try:
        with open(feedback_path, 'w') as f:
            f.write("=== Feedback ===\n\nRepetitive Words:\n")
            for word, count in repetitive.items():
                f.write(f"{word}: {count}\n")
            f.write("\nFiller Words:\n")
            for word, count in filler.items():
                f.write(f"{word}: {count}\n")
            f.write(f"\nTotal Word Count: {total}\n")
        logging.info("Feedback saved.")
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
        logging.info("Summary saved.")
    except Exception as e:
        logging.error(f"Error saving summary: {e}")

@app.route('/')
def home():
    return "GuidPro server is running! Use POST /upload to send audio."

@app.route('/upload', methods=['POST'])
def upload_audio():
    try:
        file = request.files['file']
        file.save(raw_audio_path)
        logging.info("Audio file uploaded.")
        convert_to_wav()
        transcribe_audio()
        return jsonify({"message": "Audio processed successfully."})
    except Exception as e:
        logging.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ New route to receive streaming chunks from ESP32
@app.route('/stream', methods=['POST'])
def stream():
    chunk = request.data
    with open(raw_audio_path, 'ab') as f:
        f.write(chunk)
    logging.info(f"Received chunk of size: {len(chunk)} bytes")
    return 'OK'

@app.route('/transcription.txt')
def serve_transcription():
    return serve_file(transcription_path, "Transcription not found.")

@app.route('/feedback.txt')
def serve_feedback():
    return serve_file(feedback_path, "Feedback not found.")

@app.route('/summary.txt')
def serve_summary():
    return serve_file(summary_path, "Summary not found.")

def serve_file(path, error_message):
    if os.path.exists(path):
        return send_file(path, as_attachment=False)
    else:
        abort(404, description=error_message)

@app.route('/stream', methods=['POST'])
def stream_audio():
    try:
        chunk = request.data
        with open('temp.raw', 'ab') as f:
            f.write(chunk)
        logging.info(f"Received chunk of size: {len(chunk)} bytes")
        return 'OK'
    except Exception as e:
        logging.error(f"Stream error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/process', methods=['POST'])
def process_audio():
    try:
        # Rename temp.raw to recorded_audio.raw
        if os.path.exists('temp.raw'):
            os.rename('temp.raw', raw_audio_path)
            logging.info("Accumulated audio file ready, starting processing...")
            convert_to_wav()
            transcribe_audio()
            return jsonify({"message": "Audio processed successfully."})
        else:
            return jsonify({"error": "No audio data to process."}), 400
    except Exception as e:
        logging.error(f"Process error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
