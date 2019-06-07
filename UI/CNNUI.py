import keras
from extract.SongFeatureExtractor import SongFeatureExtractor
import numpy as np


class CNN_UI:
    def __init__(self):
        self.genres = ['Folk', 'Hip-Hop', 'Pop', 'Rock', 'Instrumental']
        self.PATH_TO_CNN = r"..\saved_models\10\model-acc 0.7653- loss 0.7531.h5"
        self.extractor = SongFeatureExtractor()
        self.model = keras.models.load_model(self.PATH_TO_CNN)

    def find_songs_genre(self, path):
        predictions = np.zeros(len(self.genres))
        for mfcc in self.extractor.extract_song_features_10s_as_long_as_possbile(path):
            mfcc = np.array(mfcc)
            mfcc = mfcc.reshape(1, 20, 430, 1)
            prediction = self.model.predict(mfcc)
            prediction = np.array(prediction)
            prediction = prediction.reshape(1, 5)
            predictions = prediction + predictions

        return predictions[0]



