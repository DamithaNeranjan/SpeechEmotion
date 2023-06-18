import os

import librosa
import pickle
import soundfile
import glob

import numpy as np

from sklearn.model_selection import train_test_split

# from train_model import extract_feature

# load the model from disk
# filename = 'finalized_model.sav'
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)

# # Writing different model files to file
# with open('modelForPrediction1.sav', 'wb') as f:
#     pickle.dump(model, f)

# Emotions in the RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Emotions to observe
observed_emotions = ['angry',
                     'happy',
                     'fearful',
                     'neutral',
                     'surprised',
                     'sad']

emotionWeightages = {
    "angry": 0.55,
    "happy": 1.58,
    "fearful": 0.78,
    "neutral": 1,
    "surprise": 0.63,
    "sad": 0.45,
}

audioScore = 0.0
finalAudioScore = 0.0
audioFileCount = 0


# Load the data and extract features for each sound file
# def load_data(test_size=0.2):
#     x, y = [], []
#     for file in glob.glob(
#             "Actor_*/*.wav"):
#         file_name = os.path.basename(file)
#         emotion = emotions[file_name.split("-")[2]]
#         if emotion not in observed_emotions:
#             continue
#         feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
#         x.append(feature)
#         y.append(emotion)
#     return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result


filename = 'modelForPrediction1.sav'
loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage

# Set the path to the folder containing the WAV files
folder_path = "D:\\L4S1\\Research Project\\Final\\Candidate Videos\\6 Candidates Interview audio\\Candidate2\\Converted"

# Iterate over each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".wav"):
        # Construct the full path to the WAV file
        file_path = os.path.join(folder_path, file_name)

        # Extract features for the current WAV file
        features = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
        features = features.reshape(1, -1)

        # Make a prediction using the loaded model
        prediction = loaded_model.predict(features)

        audioFileCount = audioFileCount + 1

        if prediction == 'calm':
            prediction = ['neutral']
        elif prediction == 'disgust':
            prediction = ['fearful']

        if prediction == 'angry':
            audioScore = audioScore + emotionWeightages['angry']
            print('angry')
        elif prediction == 'fearful':
            audioScore = audioScore + emotionWeightages['fearful']
            print('fearful')
        elif prediction == 'happy':
            audioScore = audioScore + emotionWeightages['happy']
            print('happy')
        elif prediction == 'neutral' or prediction == ['neutral']:
            audioScore = audioScore + emotionWeightages['neutral']
            print('neutral')
        elif prediction == 'sad':
            audioScore = audioScore + emotionWeightages['sad']
        elif prediction == 'surprise':
            audioScore = audioScore + emotionWeightages['surprise']

        print(f"File: {file_name}, Prediction: {prediction}, Current Score: {audioScore}")

print(f"Before Final Score: {round(audioScore, 2)}")

finalAudioScore = (audioScore/audioFileCount)*100

print(f"Calculated Final Score: {round(finalAudioScore, 2)}")

# feature1 = extract_feature(
#     # "D:\\L4S1\\Research Project\\Interim\\Video Sample\\DamithaAudio1.mp3",
#     # "speech_audio/DamithaAudio1Mono.wav",
#     "D:\\L4S1\\Research Project\\Final\\Candidate Videos\\6 Candidates Interview audio\\Candidate6\\Converted\\66.wav",
#     # "speech_audio/Actor_01/03-01-01-01-01-01-01.wav",
#     # "speech_audio/03-01-01-01-01-01-01.wav",
#     mfcc=True, chroma=True, mel=True)
#
# feature1 = feature1.reshape(1, -1)
#
# prediction = loaded_model.predict(feature1)
#
# if prediction == 'calm':
#     prediction = 'neutral'
# elif prediction == 'disgust':
#     prediction = 'fearful'
#
# print(prediction)
