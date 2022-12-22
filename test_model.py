import pickle

from train_model import extract_feature

# load the model from disk
# filename = 'finalized_model.sav'
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)

# # Writing different model files to file
# with open('modelForPrediction1.sav', 'wb') as f:
#     pickle.dump(model, f)

filename = 'modelForPrediction1.sav'
loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage

feature = extract_feature(
    "Actor_01/03-01-01-01-01-01-01.wav",
    mfcc=True, chroma=True, mel=True)

feature = feature.reshape(1, -1)

prediction = loaded_model.predict(feature)

print(prediction)
