import numpy as np
import keras
from read.SongFeatureMFCCReader import ReadSongFeatureData
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from extract.SongFeatureExtractor import SongFeatureExtractor
from config import Config

class TestModelAccuracy:
    def __init__(self, model_path):

        self.genres = ['Folk', 'Hip-Hop', 'Pop', 'Rock', 'Instrumental']
        self.model = keras.models.load_model(model_path)
        self.reader = ReadSongFeatureData()
        self.extractor = SongFeatureExtractor()



    def test_model_against_longer_mfcc(self, testing_function):
        pred = []

        mfcc, labels = self.reader.read_all_genres_json(self.genres, Config.PATH_TO_EXTRACTED_TESTING_SONG_MFCCS)
        labels = list(labels)
        total = 0
        true = 0

        for single_mfcc in mfcc:
                total += 1
                if testing_function(single_mfcc, labels[mfcc.index(single_mfcc)], pred):
                    true += 1

        return pred



    def test_model_30s_clips_for_10s_model_sum(self, mfcc, label, pred):

        predictions = np.zeros(len(self.genres))
        for small_mfcc in self.extractor.make_30s_MFCC_to_10s_MFCCs(mfcc):
            small_mfcc = np.array(small_mfcc)

            small_mfcc = small_mfcc.reshape(1, 20, 430, 1)
            prediction = self.model.predict(small_mfcc)
            prediction = np.array(prediction)
            prediction = prediction.reshape(1, 5)
            predictions = prediction + predictions


        pred.append(np.argmax(predictions))
        return self.genres.index(self.label_to_genre(label)) == np.argmax(predictions[0])


    def test_model_30s_clips_for_10s_model_vote(self, mfcc, label, pred):

        predictions = np.zeros(len(self.genres))
        for small_mfcc in self.extractor.make_30s_MFCC_to_10s_MFCCs(mfcc):
            small_mfcc = np.array(small_mfcc)

            small_mfcc = small_mfcc.reshape(1, 20, 430, 1)
            prediction = self.model.predict(small_mfcc)
            prediction = np.array(prediction)
            prediction = prediction.reshape(1, 5)
            predictions[np.argmax(prediction)] += 1

        pred.append(np.argmax(predictions))
        return self.genres.index(self.label_to_genre(label)) == np.argmax(predictions[0])

    def test_model_30s_clips_for_30s_model(self, small_mfcc, label):

        predictions = np.zeros(len(self.genres))

        small_mfcc = np.array(small_mfcc)

        small_mfcc = small_mfcc.reshape(1, 20, 1291, 1)
        prediction = self.model.predict(small_mfcc)
        prediction = np.array(prediction)
        prediction = prediction.reshape(1, 5)
        predictions = prediction + predictions

        return self.genres.index(self.label_to_genre(label)) == np.argmax(predictions[0])


    def label_to_genre(self, label):

        for i in range(len(label)):
            if label[i] == 1:
                return self.genres[i]
        print("ERROR one hot encoding is false")


    def confusion_matrix_10s(self, comparing_fuction):
        _, labels_one_hot = self.reader.read_all_genres_json(self.genres, Config.PATH_TO_EXTRACTED_TESTING_SONG_MFCCS)


        pred = self.test_model_against_longer_mfcc(comparing_fuction)

        self.print_information_about_model(labels_one_hot, pred)

    def print_information_about_model(self, labels_one_hot, pred):
        labels = []
        for i in labels_one_hot:
            labels.append(np.argmax(i))
        cm = confusion_matrix(labels, pred)
        print(cm)
        sum_of_col = 0
        for i in range(len(cm)):
            sum_of_col += cm[i][i] / self.add_list([cm[0][i], cm[1][i], cm[2][i], cm[3][i], cm[4][i]])
            print(cm[i][i] / self.add_list([cm[0][i], cm[1][i], cm[2][i], cm[3][i], cm[4][i]]))
        print()
        print(sum_of_col / 5)
        sum_of_row = 0
        for i in range(len(cm)):
            sum_of_row += cm[i][i] / self.add_list(cm[i])
            print(cm[i][i] / self.add_list(cm[i]))
        print()
        print(sum_of_row / 5)
        print(classification_report(labels, pred))

    def add_list(self, l):
        sum = 0
        for i in l:
            sum += i
        return sum

    def confusion_matrix_30s(self):
        mfcc, labels_one_hot = self.reader.read_all_genres_json(self.genres, Config.PATH_TO_EXTRACTED_TESTING_SONG_MFCCS)
        print(np.shape(mfcc))

        mfcc = np.array(mfcc)
        mfcc = mfcc.reshape(500, 20, 1291, 1)
        pred = self.model.predict_classes(mfcc, batch_size=32)
        self.print_information_about_model(labels_one_hot, pred)


if __name__=="__main__":
    test = TestModelAccuracy(r"..\saved_models\10\model-acc 0.7653- loss 0.7531.h5")
    test.confusion_matrix_10s(test.test_model_30s_clips_for_10s_model_sum)
    test.confusion_matrix_10s(test.test_model_30s_clips_for_10s_model_vote)
    test = TestModelAccuracy(r"..\saved_models\30\model-acc 0.6160- loss 1.1028.h5")
    test.confusion_matrix_30s()




