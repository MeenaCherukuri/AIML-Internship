# 🎙️ Human Emotion Detection from Voice

Detect the **emotion behind a speaker’s voice** using audio recordings and machine learning. This project uses the **RAVDESS dataset**, audio processing libraries like **Librosa**, and **Scikit-learn** for training, along with a **Streamlit UI** for live interaction.

---

## 🚀 Project Overview

This project builds an intelligent system that can identify human emotions like **happy**, **angry**, **sad**, and more from voice recordings using machine learning. It extracts **audio features** and trains a classifier to predict emotional states in real-time or from uploaded `.wav` files.

---

## 🎯 Objective

To detect the **emotion of a speaker** using audio input by:
- Extracting meaningful features from voice
- Training an emotion classifier
- Building a UI for real-time predictions

---

## 🧰 Tools & Technologies

| Category       | Technology      |
|----------------|------------------|
| Programming    | Python           |
| Audio Features | Librosa          |
| ML Models      | Scikit-learn     |
| UI             | Streamlit        |
| Dataset        | RAVDESS          |

---


---

## 🗂️ Dataset

- **Name**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Download**: [Kaggle RAVDESS Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

---

## ⚙️ How It Works

### ✅ Feature Extraction

Each audio file is processed to extract meaningful patterns using:

- 🎵 **MFCC (Mel-Frequency Cepstral Coefficients)**: Captures the timbral texture of the voice.
- 🎼 **Chroma Features**: Captures pitch class content (related to musical elements).
- 📊 **Mel Spectrogram**: Represents the energy of frequencies over time.

These features are combined into a single feature vector per audio sample and used for training.


### ✅ Classifier
- Trained using **Random Forest** or **SVM**

### ✅ Streamlit UI
- Record or upload voice
- Predict emotion
- Display result in UI

---

## 📈 Model Performance

| Metric     | Value    |
|------------|----------|
| Accuracy   | ~85–90%  |
| Classes    | Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised |

---

## 🧪 How to Run Locally (Colab + Streamlit)

### 1️⃣ Setup in Google Colab

    ```python
     # Install dependencies
     !pip install librosa soundfile scikit-learn kaggle resampy
# Upload kaggle.json and download dataset
from google.colab import files
files.upload()  # Upload your kaggle.json

### 2️⃣ Train the Model

    ```bash
    python train_model.py

### 3️⃣ Run the Streamlit App
    ```bash
    streamlit run app.py

## 🌐 Streamlit Web App Features

- 🎤 **Upload your voice or record live**
- 🧠 **Get predicted emotion instantly**
- 📊 **(Optional) Visualize emotion trend over time**
  

## 📝 Future Enhancements

- 🎧 **Real-time microphone input**
- 📈 **Emotion trend chart**
- 🔁 **Use deep learning models (CNN, LSTM)**
- 📱 **Mobile/web deployment**

## 🔍 Project Motivation

In human communication, tone and emotion play a crucial role alongside the spoken words. With the rise of AI and voice-driven interfaces, emotion detection from speech has become essential in fields like:

- 🎧 Voice assistants (e.g., Alexa, Siri)
- 🧠 Mental health analysis
- 🎓 E-learning engagement tracking
- 🎮 Emotion-aware games
- 🤖 Human-robot interaction

This project aims to bridge that emotional gap by training a machine learning model to recognize emotions from audio using classical ML techniques.

## 🔄 Possible Model Improvements

- 🔍 Try different classifiers (e.g., XGBoost, Gradient Boosting)
- 📦 Perform hyperparameter tuning using GridSearchCV or Optuna
- 🧠 Replace classical ML with deep learning (CNNs, RNNs, LSTMs)
- 🏷️ Use data augmentation techniques (noise addition, pitch shift) to increase robustness
- 🔁 Use transfer learning from pretrained audio models (e.g., YAMNet, Wav2Vec)

## 🙌 Acknowledgements

- 🎵 [RAVDESS Dataset (Official)](https://zenodo.org/record/1188976)
- 📂 [Kaggle RAVDESS Source](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)


## 🧑‍💻 How to Contribute

If you'd like to improve this project, you're welcome to contribute! Here's how:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-new`)
3. Make your changes and commit (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-new`)
5. Create a new Pull Request


## 📡 Real-Time Applications

- 🧠 **Mental Health Monitoring**: Detect emotional trends over time from call center data or therapy sessions.
- 📞 **Call Center Optimization**: Recognize customer frustration or satisfaction during live calls.
- 🗣️ **Virtual Assistant Personalization**: Adjust responses based on user's emotional state.
- 📚 **EdTech**: Measure student engagement during online learning.

## 🧪 Suggested Advanced Features

- 🔴 Real-time live microphone-based emotion detection (using WebRTC or `sounddevice`)
- 🔉 Multi-language emotion support (train on multilingual audio datasets)
- 📱 Deploy as a mobile app using Streamlit + WebView or React Native
- 🧵 Add audio visualization using `matplotlib` or `plotly` in Streamlit
- 🗃️ Add a database layer (e.g., SQLite/MongoDB) to store user audio and results

## Results Snapshot

| Emotion     | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Happy       | 0.89      | 0.86   | 0.87     |
| Angry       | 0.91      | 0.88   | 0.89     |
| Sad         | 0.88      | 0.87   | 0.87     |
| Calm        | 0.85      | 0.84   | 0.84     |
| **Average** | **0.88**  | **0.86** | **0.87** |



