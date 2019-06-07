from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation, BatchNormalization
from keras.regularizers import l2
import numpy as np
import random
from read.SongFeatureMFCCReader import ReadSongFeatureData
from keras.optimizers import Adam



class CNN:

    def __init__(self):
        self.genres = ['Folk', 'Hip-Hop', 'Pop', 'Rock', 'Instrumental']
        self.regularizer = l2(1e-5)
        self.model = self.get_CNN((20, 430, 1))

    def get_CNN(self, shape):
        model = Sequential()
        model.add(Conv2D(input_shape=shape, kernel_size=(3, 3), filters=24, kernel_regularizer=self.regularizer))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))

        model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=self.regularizer))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Conv2D(48, (3, 3), padding='same', kernel_regularizer=self.regularizer))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(62, (3, 3), padding='same', kernel_regularizer=self.regularizer))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=self.regularizer))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=self.regularizer))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
        model.add(Dropout(0.3))



        model.add(Flatten())
        model.add(BatchNormalization())

        model.add(Dense(512, kernel_regularizer=self.regularizer))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(Dense(256, kernel_regularizer=self.regularizer))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(len(self.genres), activation='softmax'))

        optimizer = Adam(lr=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return model
    def get_data(self):
        mfcc, label = self.get_mfcc_data_for_all_genres()

        mfcc, label = self.randomise_mfcc_label_order(label, mfcc)

        test_input, test_labels, train_input, train_labels = self.split_training_testing_data(mfcc, label)

        return train_input, train_labels, test_input, test_labels

    def randomise_mfcc_label_order(self, labels, mfcc):

        labels = list(labels)
        mfcc = list(mfcc)
        data = list(zip(labels, mfcc))
        random.shuffle(data)
        labels_shuffled, mfcc_shuffled = [], []
        labels_shuffled[:], mfcc_shuffled[:] = zip(*data)

        return mfcc_shuffled, labels_shuffled


    def get_mfcc_data_for_all_genres(self):
        reader = ReadSongFeatureData()
        mfcc, labels = reader.read_all_genres_json(self.genres)

        return mfcc, labels

    def split_training_testing_data(self, mfcc, labels):
        train_split = 0.9

        index = int(len(labels) * train_split)

        train_input = mfcc[:index]
        train_labels = labels[:index]

        test_input = mfcc[index:]
        test_labels = labels[index:]

        test_input = np.array(test_input)
        train_input = np.array(train_input)

        test_input = test_input.reshape(len(test_input), 20, 430, 1)
        train_input = train_input.reshape(len(train_input), 20, 430, 1)

        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        return test_input, test_labels, train_input, train_labels

    def fit_model(self):
        train_input, train_labels, test_input, test_labels = self.get_data()
        print('shape of train input ', np.shape(train_input))
        print('shape of train_labels ', np.shape(train_labels))
        print('shape of test_input ', np.shape(test_input))
        print('shape of test_labels ', np.shape(test_labels))

        self.model.fit(train_input, train_labels, epochs=90, batch_size=32, validation_split=0.1)
        loss, acc = self.model.evaluate(test_input, test_labels, batch_size=32)

        print("Loss: %.4f and Acc: %.4f" % (loss, acc))
        return self.model, loss, acc

if __name__=="__main__":
    cnn = CNN()
    cnn.fit_model()
