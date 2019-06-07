import librosa as ls
from config import Config
import numpy as np
import json
import os

class SongFeatureExtractor:


    def __init__(self):
        np.set_printoptions(threshold=10000000)



    def extract_song_features_10_sec_parts(self, path):
            y, sr = ls.load(path)
            mfcc = ls.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc /= np.amax(np.absolute(mfcc))
            print(np.shape(mfcc))
            if np.shape(mfcc) >= (20, 1291):
                mfcc1, mfcc2, mfcc3 = self.make_30s_MFCC_to_10s_MFCCs(mfcc)

                return mfcc1, mfcc2, mfcc3

    def make_30s_MFCC_to_10s_MFCCs(self, mfcc):
        mfcc = np.array(mfcc)
        mfcc1 = mfcc[:, 0:430]
        mfcc2 = mfcc[:, 430:860]
        mfcc3 = mfcc[:, 860:1290]
        mfcc1 = mfcc1.tolist()
        mfcc2 = mfcc2.tolist()
        mfcc3 = mfcc3.tolist()
        return mfcc1, mfcc2, mfcc3

    def extract_song_features_30_sec_parts(self, path):
        y, sr = ls.load(path)
        mfcc = ls.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc /= np.amax(np.absolute(mfcc))
        print(np.shape(mfcc))
        if np.shape(mfcc) >= (20, 1291):
            mfcc = np.array(mfcc)
            mfcc = mfcc[:, :1291]
            mfcc = mfcc.tolist()
            return mfcc
    def extract_song_features_10s_as_long_as_possbile(self, path):
        mfccs = []

        y, sr = ls.load(path)
        mfcc = ls.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc /= np.amax(np.absolute(mfcc))
        if np.shape(mfcc) >= (20, 430):
            counter = int(len(mfcc[0]) / 430)
            mfcc = np.array(mfcc)
            for i in range(counter):
                mfcc1 = mfcc[:, i * 430: (i+1) * 430]
                mfcc1 = mfcc1.tolist()
                mfccs.append(mfcc1)

            return mfccs
    def find_path_to_song_file_from_song_id(self, song_id):
        return Config.PATH_TO_FMA + "\{folder}\{file}.wav".format(folder=song_id[0:3], file=song_id)


    def find_path_to_song_file_from_song_id_test_data(self, song_id):
        return Config.PATH_TO_TEST_DATA + "\{folder}\{file}.wav".format(folder=song_id[0:3], file=song_id)




    def write_to_json(self, mfcc, genre):

        data = [genre, mfcc]
        with open("%s.json" % genre, "w") as write_file:
            json.dump(data, write_file)


    def read_json_file(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data[1]


    def extract_features_from_single_genre(self, path, extraction_function):
        songs = list(self.read_json_file(path))
        mfccs = []

        print("total songs %d" % len(songs))
        for song in songs:
            exists = os.path.isfile(self.find_path_to_song_file_from_song_id(song))
            if exists:
                extraction_function(mfccs, song)
        return mfccs

    def extration_10_sec(self, mfccs, song):
        mfcc = self.extract_song_features_10_sec_parts(self.find_path_to_song_file_from_song_id(song))
        if mfcc is not None:
            for one_mfcc in mfcc:
                mfccs.append(one_mfcc)
        print(len(mfccs) / 3)

    def extration_30_sec(self, mfccs, song):
        mfcc = self.extract_song_features_30_sec_parts(self.find_path_to_song_file_from_song_id(song))
        if mfcc is not None:
            mfccs.append(mfcc)
        print(len(mfccs))

    def write_json_file_for_single_genre(self, genre, extraction_function):
        self.write_to_json(self.extract_features_from_single_genre(r"%s\%s.json" % (Config.PATH_TO_SONG_ID_DATA, genre), extraction_function), genre)

    def write_json_file_for_single_test_genre(self, path, genre):
        self.write_to_json(self.extract_test_songs_features(path, 100), genre)


    def find_if_song_in_FMA_small(self, path):
        songs = list(self.read_json_file(path))

        for song in songs:
            exists = os.path.isfile(self.find_path_to_song_file_from_song_id(song))
            if exists:
                songs.remove(song)
        return songs

    def extract_test_songs_features(self, path, number_of_test_songs):
        songs = self.find_if_song_in_FMA_small(path)
        mfccs = []

        for song in songs:
            exists = os.path.isfile(self.find_path_to_song_file_from_song_id_test_data(song))
            if exists:
                mfcc = self.extract_song_features_30_sec_parts(self.find_path_to_song_file_from_song_id_test_data(song))
                if mfcc is not None:

                    mfccs.append(mfcc)
                print(len(mfccs))
                if(len(mfccs) > number_of_test_songs):
                    break


        return mfccs

if __name__=="__main__":
    tester = SongFeatureExtractor()
    tester.write_json_file_for_single_genre("Pop", tester.extration_10_sec)
    tester.write_json_file_for_single_genre("Folk", tester.extration_30_sec)
# Hip Hop 997
# Folk 1000
# Rock 999
# Pop 1000
# Instrumental 1000
