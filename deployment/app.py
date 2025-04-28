import os
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import threading
import time
from collections import Counter

# Import your inference module
from inference import process_local_media

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'gif', 'jpg', 'jpeg', 'png', 'webp', 'bmp'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create a folder for storing analysis results
os.makedirs('results', exist_ok=True)

# Emoji mappings
EMOTION_EMOJIS = {
    "anger": "😡",
    "disgust": "🤢",
    "fear": "😨",
    "joy": "😄",
    "neutral": "😐",
    "sadness": "😢",
    "surprise": "😲"
}

SENTIMENT_EMOJIS = {
    "negative": "👎",
    "neutral": "🤷",
    "positive": "👍"
}

# Dictionary to track processing status
processing_jobs = {}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_file_async(file_path, job_id):
    """Process file asynchronously and save results"""
    try:
        processing_jobs[job_id]['status'] = 'processing'
        
        # Process the file using your model
        results = process_local_media(file_path)
        
        # Calculate the dominant emotion and sentiment
        all_emotions = []
        all_sentiments = []
        
        for utt in results:
            # Get the top emotion (highest confidence)
            top_emotion = max(utt["emotions"], key=lambda x: x["confidence"])
            all_emotions.append(top_emotion["label"])
            
            # Get the top sentiment (highest confidence)
            top_sentiment = max(utt["sentiments"], key=lambda x: x["confidence"])
            all_sentiments.append(top_sentiment["label"])
        
        # Find the most common emotion and sentiment
        dominant_emotion = Counter(all_emotions).most_common(1)[0][0] if all_emotions else "neutral"
        dominant_sentiment = Counter(all_sentiments).most_common(1)[0][0] if all_sentiments else "neutral"
        
        # Save results
        result_data = {
            'results': results,
            'summary': {
                'dominant_emotion': dominant_emotion,
                'dominant_sentiment': dominant_sentiment,
                'emotion_emoji': EMOTION_EMOJIS[dominant_emotion],
                'sentiment_emoji': SENTIMENT_EMOJIS[dominant_sentiment],
                'utterance_count': len(results)
            }
        }
        
        result_file = os.path.join('results', f"{job_id}.json")
        with open(result_file, 'w') as f:
            json.dump(result_data, f)
        
        processing_jobs[job_id]['status'] = 'completed'
        
    except Exception as e:
        processing_jobs[job_id]['status'] = 'failed'
        processing_jobs[job_id]['error'] = str(e)
        print(f"Error processing file: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Generate job ID based on timestamp and filename
        job_id = f"{int(time.time())}_{filename}"
        
        # Start processing in background
        processing_jobs[job_id] = {'status': 'starting', 'filename': filename}
        thread = threading.Thread(target=process_file_async, args=(file_path, job_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({'job_id': job_id})
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/status/<job_id>')
def job_status(job_id):
    if job_id not in processing_jobs:
        return jsonify({'status': 'not_found'}), 404
    
    return jsonify({
        'status': processing_jobs[job_id]['status'],
        'error': processing_jobs[job_id].get('error', None)
    })

@app.route('/results/<job_id>')
def get_results(job_id):
    result_file = os.path.join('results', f"{job_id}.json")
    
    if not os.path.exists(result_file):
        return jsonify({'error': 'Results not found'}), 404
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    return render_template('results.html', 
                          results=results['results'],
                          summary=results['summary'],
                          emotion_emojis=EMOTION_EMOJIS,
                          sentiment_emojis=SENTIMENT_EMOJIS)

if __name__ == '__main__':
    app.run(debug=True)