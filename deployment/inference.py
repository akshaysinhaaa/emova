import torch
from models import MultimodalSentimentModel
import os
import cv2
import numpy as np
import subprocess
import torchaudio
import whisper
from transformers import AutoTokenizer
import json
import pytesseract
from PIL import Image
import imageio

EMOTION_MAP = {0: "anger", 1: "disgust", 2: "fear",
               3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}

class MediaProcessor:
    def process_media(self, file_path):
        """Process different media types (video, GIF, or static image)"""
        # Get file extension
        _, ext = os.path.splitext(file_path.lower())
        
        # Process based on file type
        if ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
            return self.process_static_image(file_path)
        elif ext == '.gif':
            return self.process_gif(file_path)
        else:  # Default to video processing
            return self.process_video(file_path)
    
    def process_static_image(self, image_path):
        """Process a static image and convert it to video frames format"""
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
                
            # Resize and normalize
            image = cv2.resize(image, (224, 224))
            image = image / 255.0
            
            # Create 30 duplicate frames (as our model expects video)
            frames = [image] * 30
            
            return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
            
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
    
    def process_gif(self, gif_path):
        try:
            # Use imageio to read GIF
            gif = imageio.mimread(gif_path)
            frames = []
            
            # Process frames
            for frame in gif[:30]:  # Limit to 30 frames
                # Convert to RGB if grayscale
                if len(frame.shape) == 2:
                    frame = np.stack([frame, frame, frame], axis=2)
                
                # Ensure we have RGB (not RGBA)
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                    
                # Resize and normalize
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)
                
            if len(frames) == 0:
                raise ValueError("No frames extracted from GIF")
                
            # Pad or truncate to exactly 30 frames
            if len(frames) < 30:
                frames += [np.zeros_like(frames[0])] * (30 - len(frames))
            else:
                frames = frames[:30]
                
            return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
            
        except Exception as e:
            raise ValueError(f"Error processing GIF: {str(e)}")
    
    def process_video(self, video_path):
        # Standard video processing
        cap = cv2.VideoCapture(video_path)
        frames = []

        if not cap.isOpened():
            raise ValueError(f"Video not found: {video_path}")

        while len(frames) < 30 and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError("No frames extracted")

        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

class AudioProcessor:
    def extract_features(self, file_path, max_length=300):
        # Skip audio extraction for images and GIFs
        _, ext = os.path.splitext(file_path.lower())
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
            return None
            
        audio_path = file_path + '.wav'

        try:
            result = subprocess.run([
                'ffmpeg',
                '-i', file_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if not os.path.exists(audio_path):
                return None

            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                hop_length=512,
                n_mels=64
            )(waveform)

            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            if mel_spec.size(2) < max_length:
                padding = max_length - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :max_length]

            os.remove(audio_path)
            return mel_spec
        except Exception:
            return None

class MediaUtteranceProcessor:
    def __init__(self):
        self.media_processor = MediaProcessor()
        self.audio_processor = AudioProcessor()

    def extract_segment(self, file_path, start_time, end_time, temp_dir="tmp"):
        # Skip segmentation for images and GIFs
        _, ext = os.path.splitext(file_path.lower())
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
            return file_path
            
        os.makedirs(temp_dir, exist_ok=True)
        segment_path = os.path.join(temp_dir, f"segment_{start_time}_{end_time}.mp4")

        subprocess.run([
            "ffmpeg", "-i", file_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",
            segment_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return segment_path

def load_model(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalSentimentModel().to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), map_location=device))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    transcriber = whisper.load_model("base", device="cpu" if device.type == "cpu" else device)

    return model, tokenizer, transcriber, device

def ocr_from_image(image_path):
    """Extract text from a static image using OCR"""
    try:
        # Read the image
        img = Image.open(image_path)
        
        # Apply OCR directly
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"Error extracting OCR from image: {e}")
        return ""

def ocr_from_frames(frames, max_frames=10):
    """Extract text from frames using OCR"""
    extracted_text = []
    
    for i, frame in enumerate(frames):
        if i >= max_frames:
            break
            
        # Convert normalized float array back to uint8 for OCR
        if isinstance(frame, np.ndarray) and frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
            
        # Convert to grayscale if colored
        if len(frame.shape) == 3 and frame.shape[2] >= 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
            
        # Apply OCR
        pil_img = Image.fromarray(gray)
        text = pytesseract.image_to_string(pil_img)
        if text.strip():
            extracted_text.append(text.strip())
            
    return " ".join(extracted_text)

def ocr_from_media(file_path):
    """Extract text from video, GIF or image using OCR"""
    _, ext = os.path.splitext(file_path.lower())
    
    # For static images, use direct OCR
    if ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
        return ocr_from_image(file_path)
    
    # For GIFs, use imageio
    if ext == '.gif':
        try:
            gif = imageio.mimread(file_path)
            return ocr_from_frames(gif)
        except Exception as e:
            print(f"Error extracting OCR from GIF: {e}")
            return ""
    
    # For videos, use OpenCV
    cap = cv2.VideoCapture(file_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count > 10:
            break
        if frame_count % 5 == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return ocr_from_frames(frames)

def predict(input_path, model, tokenizer, transcriber, device):
    utterance_processor = MediaUtteranceProcessor()
    predictions = []
    
    # Get file extension to determine media type
    _, ext = os.path.splitext(input_path.lower())
    is_static_image = ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    is_gif = ext == '.gif'
    
    # For static images and GIFs: skip audio transcription and use OCR directly
    if is_static_image or is_gif:
        media_type = "static image" if is_static_image else "GIF"
        print(f"Processing {media_type} - extracting OCR text")
        
        # Extract text via OCR
        text = ocr_from_media(input_path)
        if not text.strip():
            print(f"Warning: No text extracted from {media_type} via OCR")
            text = ""  # Empty text will still be analyzed by sentiment model

        # Process visual content
        media_frames = utterance_processor.media_processor.process_media(input_path)
        
        # Use dummy zero tensor for audio since images/GIFs don't have audio
        audio_features = torch.zeros((1, 1, 64, 300)).to(device)
        
        # Prepare text input
        text_inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        # Run inference
        with torch.inference_mode():
            outputs = model(text_inputs, media_frames.unsqueeze(0).to(device), audio_features)
            emotion_probs = torch.softmax(outputs["emotions"], dim=1)[0]
            sentiment_probs = torch.softmax(outputs["sentiments"], dim=1)[0]

            emotion_values, emotion_indices = torch.topk(emotion_probs, 3)
            sentiment_values, sentiment_indices = torch.topk(sentiment_probs, 3)

        predictions.append({
            "start_time": 0,
            "end_time": 0,
            "text": text,
            "emotions": [
                {"label": EMOTION_MAP[idx.item()], "confidence": conf.item()}
                for idx, conf in zip(emotion_indices, emotion_values)
            ],
            "sentiments": [
                {"label": SENTIMENT_MAP[idx.item()], "confidence": conf.item()}
                for idx, conf in zip(sentiment_indices, sentiment_values)
            ]
        })

        return predictions

    # For videos: try whisper transcription first, fall back to OCR
    try:
        result = transcriber.transcribe(input_path, word_timestamps=True)
        segments = result.get("segments", [])
        if not segments:
            raise Exception("No segments found")
    except Exception as e:
        print(f"Transcription failed: {e}")
        print("No audio stream or Whisper failed. Using OCR to extract text.")
        text = ocr_from_media(input_path)
        if not text.strip():
            print("Warning: No text extracted from video")
            text = ""  # Empty text will still be analyzed

        media_frames = utterance_processor.media_processor.process_media(input_path)
        audio_features = torch.zeros((1, 1, 64, 300)).to(device)  # Dummy zero audio
        
        text_inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        with torch.inference_mode():
            outputs = model(text_inputs, media_frames.unsqueeze(0).to(device), audio_features)
            emotion_probs = torch.softmax(outputs["emotions"], dim=1)[0]
            sentiment_probs = torch.softmax(outputs["sentiments"], dim=1)[0]

            emotion_values, emotion_indices = torch.topk(emotion_probs, 3)
            sentiment_values, sentiment_indices = torch.topk(sentiment_probs, 3)

        predictions.append({
            "start_time": 0,
            "end_time": 0,
            "text": text,
            "emotions": [
                {"label": EMOTION_MAP[idx.item()], "confidence": conf.item()}
                for idx, conf in zip(emotion_indices, emotion_values)
            ],
            "sentiments": [
                {"label": SENTIMENT_MAP[idx.item()], "confidence": conf.item()}
                for idx, conf in zip(sentiment_indices, sentiment_values)
            ]
        })

        return predictions

    # Process each segment for videos with audio
    for segment in segments:
        try:
            segment_path = utterance_processor.extract_segment(input_path, segment["start"], segment["end"])
            media_frames = utterance_processor.media_processor.process_media(segment_path)
            audio_features = utterance_processor.audio_processor.extract_features(segment_path)

            if audio_features is None:
                audio_features = torch.zeros((1, 64, 300))

            text_inputs = tokenizer(segment["text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            media_frames = media_frames.unsqueeze(0).to(device)
            audio_features = audio_features.unsqueeze(0).to(device)

            with torch.inference_mode():
                outputs = model(text_inputs, media_frames, audio_features)
                emotion_probs = torch.softmax(outputs["emotions"], dim=1)[0]
                sentiment_probs = torch.softmax(outputs["sentiments"], dim=1)[0]

                emotion_values, emotion_indices = torch.topk(emotion_probs, 3)
                sentiment_values, sentiment_indices = torch.topk(sentiment_probs, 3)

            predictions.append({
                "start_time": segment["start"],
                "end_time": segment["end"],
                "text": segment["text"],
                "emotions": [
                    {"label": EMOTION_MAP[idx.item()], "confidence": conf.item()}
                    for idx, conf in zip(emotion_indices, emotion_values)
                ],
                "sentiments": [
                    {"label": SENTIMENT_MAP[idx.item()], "confidence": conf.item()}
                    for idx, conf in zip(sentiment_indices, sentiment_values)
                ]
            })

        except Exception as e:
            print(f"Segment processing failed: {e}")

        finally:
            if os.path.exists(segment_path) and segment_path != input_path:
                os.remove(segment_path)

    return predictions

def process_local_media(input_path, model_dir="model_normalized"):
    print(f"Processing media file: {input_path}")
    model, tokenizer, transcriber, device = load_model(model_dir)
    results = predict(input_path, model, tokenizer, transcriber, device)

    for utt in results:
        print(f"\n Utterance: {utt['text']}")
        if utt['start_time'] != utt['end_time']:
            print(f"Time: {utt['start_time']:.2f}s - {utt['end_time']:.2f}s")
        print("Emotions:")
        for emo in utt["emotions"]:
            print(f"  - {emo['label']}: {emo['confidence']:.2f}")
        print("Sentiments:")
        for sent in utt["sentiments"]:
            print(f"  - {sent['label']}: {sent['confidence']:.2f}")
        print("-" * 40)
    
    return results

import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <media_path>")
        print("Supports: .mp4, .gif, .jpg, .jpeg, .png, .webp, .bmp")
        sys.exit(1)

    input_path = sys.argv[1]
    process_local_media(input_path)