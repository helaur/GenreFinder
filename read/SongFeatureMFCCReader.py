import json
import numpy as np
from keras.utils.np_utils import to_categorical
from config import Config

class ReadSongFeatureData:

    def read_json_file(self, path, genre):

        labels = []
        with open(path, 'r') as file:
            data = json.load(file)

            for i in data[1]:
                labels.append(genre)

        return data[1], labels

    def read_all_genres_json(self, genres, path=Config.PATH_TO_EXTRACTED_SONG_MFCC):
        all_mfccs = []
        all_labels = []

        for genre in genres:
            single_genre_mfcc, single_genre_labels = self.read_json_file(path + "\%s.json" % genre, genre)
            all_mfccs.extend(single_genre_mfcc)
            all_labels.extend(single_genre_labels)

        label_ids, label_row_ids = np.unique(all_labels, return_inverse=True)
        onehot_encoding_labels = to_categorical(label_row_ids, len(label_ids))

        return all_mfccs, onehot_encoding_labels

if __name__ == "__main__":

    reader = ReadSongFeatureData()

