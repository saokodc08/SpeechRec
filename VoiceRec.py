import os
import librosa
import IPython.display as ipd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.models import load_model
import sounddevice as sd
import soundfile as sf
import random

warnings.filterwarnings("ignore")

# samples, sample_rate = librosa.load(
#     'D:/PyCharm Community Edition 2021.2.2/Project/input/tensorflow-speech-recognition-challenge/train/audio\yes/0a7c2a8d_nohash_0.wav', sr=16000)
# ipd.Audio(samples, rate=sample_rate)
# samples = librosa.resample(samples, sample_rate, 8000)
# ipd.Audio(samples, rate=8000)

# Get labels
train_audio_path = 'D:/PyCharm Community Edition 2021.2.2/Project/input/tensorflow-speech-recognition-challenge/train/audio/'
labels = os.listdir(train_audio_path)
labels = ["yes", "no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero"]
all_label = labels
# Removing shorter commands of less than 1 second and resample them to 8000 hz
# since most of the speech related frequencies are present in 8000z
all_wave = []
all_label = []
for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '/' + label) if
             f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr=16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if (len(samples) == 8000):
            all_wave.append(samples)
            all_label.append(label)
    print(label, " done")

# Convert the output labels to integer encoded
le = LabelEncoder()
y = le.fit_transform(all_label)
classes = list(le.classes_)
# Convert the integer encoded labels to a one-hot vector since it is a multi-classification problem
y = np_utils.to_categorical(y, num_classes=len(labels))
# Reshape the 2D array to 3D since the input to the conv1d must be a 3D array
all_wave = np.array(all_wave).reshape(-1, 8000, 1)
# Train the model on 80% of the data and validate on the remaining 20%
x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave), np.array(y), stratify=y, test_size=0.2, random_state=777, shuffle=True)
# build the speech-to-text model using conv1d.
# Conv1d is a convolutional neural network which performs the convolution along only one dimension
# Implement the model using Keras functional API
K.clear_session()
inputs = Input(shape=(8000, 1))
# First Conv1D layer
conv = Conv1D(8, 13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)
# Second Conv1D layer
conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)
# Third Conv1D layer
conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)
# Fourth Conv1D layer
conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)
# Flatten layer
conv = Flatten()(conv)
# Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)
# Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)
outputs = Dense(len(labels), activation='softmax')(conv)
model = Model(inputs, outputs)
model.summary()
# Define the loss function to be categorical cross-entropy since it is a multi-classification problem
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Early stopping and model checkpoints are the callbacks to stop training
# the neural network at the right time and to save the best model after every epoch
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
mc = ModelCheckpoint('SpeechRecModel.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Train the model on a batch size of 32 and evaluate the performance on the holdout set
history = model.fit(x_tr, y_tr, epochs=100, callbacks=[es, mc], batch_size=32, validation_data=(x_val, y_val))

# Loading the best model
model = load_model('SpeechRecModel.hdf5')


# Define the function that predicts text for the given audio
def predict(audio):
    prob = model.predict(audio.reshape(1, 8000, 1))
    index = np.argmax(prob[0])
    return classes[index]


# Random prediction
index = random.randint(0, len(x_val) - 1)
samples = x_val[index].ravel()
print("Audio:", classes[np.argmax(y_val[index])])
ipd.Audio(samples, rate=8000)
print("Text:", predict(samples))

# Record voice
samplerate = 16000
duration = 1  # seconds
filename = 'voice.wav'
print("start")
mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,channels=1, blocking=True)
print("end")
sd.wait()
sf.write(filename, mydata, samplerate)

# Read the saved voice command and convert it to text
os.listdir('D:/PyCharm Community Edition 2021.2.2/Project')
filepath = 'D:/PyCharm Community Edition 2021.2.2/Project'
# reading the voice commands
samples, sample_rate = librosa.load(filepath + '/' + 'voice.wav', sr=16000)
samples = librosa.resample(samples, sample_rate, 8000)
ipd.Audio(samples, rate=8000)
# converting voice commands to text
predict(samples)
