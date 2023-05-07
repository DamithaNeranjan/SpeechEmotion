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
observed_emotions = ['calm', 'happy', 'fearful', 'neutral', 'surprised']


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
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result


filename = 'modelForPrediction1.sav'
loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage

feature1 = extract_feature(
    # "D:\\L4S1\\Research Project\\Interim\\Video Sample\\DamithaAudio1.mp3",
    "speech_audio/DamithaAudio1Mono.wav",
    # "speech_audio/Actor_01/03-01-01-01-01-01-01.wav",
    # "speech_audio/03-01-01-01-01-01-01.wav",
    mfcc=True, chroma=True, mel=True)

feature1 = feature1.reshape(1, -1)

prediction = loaded_model.predict(feature1)

print(prediction)
