# Multi-Task Learning for Conversational Emotion and Sentiment Recognition

A deep learning framework designed for **emotion and sentiment recognition** using **text**, **audio**, and **video** modalities. This project leverages the **MELD (Multimodal EmotionLines Dataset)** to train a robust and flexible model that reflects human communication more accurately than unimodal models.

---

## 📦 Dataset: MELD

**Multimodal EmotionLines Dataset (MELD)** is a large-scale, multi-party conversation dataset derived from the TV series *Friends*. It provides aligned and synchronized **text**, **audio**, and **video** data, annotated with both **emotion** and **sentiment** labels.

- **Modalities**:  
  - `Text`: Dialogues (utterances)  
  - `Audio`: Speaker voice tone  
  - `Video`: Speaker facial expressions and posture

- **Emotion Labels**:  
  - Anger  
  - Disgust  
  - Fear  
  - Joy  
  - Neutral  
  - Sadness  
  - Surprise  

- **Sentiment Labels**:  
  - Positive  
  - Negative  
  - Neutral  

🔗 [MELD Dataset GitHub](https://github.com/declare-lab/MELD)

---

## 🧠 Model Architecture

The model is **modular** and allows training on individual or fused modalities: `Text`, `Audio`, and `Video`. It is designed to perform well when one or more modalities are missing or unavailable.

### 🔹 Individual Modality Encoders

| Modality | Model Used         | Preprocessing                  |
|----------|--------------------|--------------------------------|
| Text     | BERT               | Tokenization, Padding          |
| Audio    | CNN                | MFCC / Log-Mel Spectrogram     |
| Video    | ResNet18 / 3D-CNN  | Face Extraction, Frame Sampling|

### 🔹 Multimodal Fusion Strategy

- Concatenation of latent vectors from each modality  
- Optional **attention mechanism** to weight more informative modalities  
- Final **Fully Connected Layers** leading to classification head (Softmax)

```
             ┌────────────┐    ┌────────────┐     ┌────────────┐
             │   Text     │    │   Audio    │     │   Video    │
             └────┬───────┘    └── ┬─────── ┘      └────┬───────┘
                  │                │                   │
                 BERT             CNN            3D CNN / ResNet
                  │                │                   │
                  └────────────┬───┴────┬──────────────┘
                               │ Fusion │
                               └────┬───┘
                            Fully Connected
                               Softmax
```

---

## 🧪 Training Details

- **Optimizer**: AdamW  
- **Scheduler**: ReduceLROnPlateau  
- **Loss Function**:  
  - CrossEntropyLoss for multiclass emotion classification  
  - Label Smoothing (0.1) to prevent overconfidence  
- **Regularization**:  
  - Dropout in FC layers (0.3–0.5)  
  - Early Stopping based on validation loss  
- **Batch Size**: 16–32  
- **Epochs**: 15–25  

### 🧵 Hyperparameter Tuning

- Performed manually (grid search) on:
  - Learning rate (1e-3 to 1e-5)  
  - Hidden layer sizes  
  - Dropout rates  
  - Fusion strategies (early vs late fusion)

---

## 📈 Performance Snapshot

| Configuration          | Emo Precision | Emo Acc. | Sen Precision | Sen Acc. |
|------------------------|---------------|----------|---------------|----------|
| Fused Model            | 65.00%        | 75.00%   | 85.00%        | 95.00%   |

---

## 🧑‍💻 Author

**Akshay Sinha, Gauri Saksena, Yash Chandel**  
_Deep Learning | Multimodal AI | Emotion Recognition_
