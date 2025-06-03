# Train model to classify Vietnamese money

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.applications.resnet50 import ResNet50, preprocess_input
from keras._tf_keras.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import random
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

raw_folder = "data/"

def save_data(raw_folder=raw_folder):

    destination_size = (128, 128)
    print("Processing")

    pixels = []
    labels = []

    # Loop all sub-folder
    for folder in os.listdir(raw_folder):
        print("Folder: ", folder)
        # Loop all file into sub-folders
        for file in os.listdir(raw_folder + folder):
            print("File: ", file)
            pixels.append(cv2.resize(cv2.imread(raw_folder + folder + "/" + file), dsize=destination_size))
            labels.append(folder)

    pixels = np.array(pixels)
    labels = np.array(labels).reshape(-1, 1)

    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)
    print(labels)

    file = open('pixels.data', 'wb')
    pickle.dump((pixels, labels), file)
    file.close()

    return

# Run once
#save_data()

def load_data():
    file = open('pixels.data', 'rb')

    (pixels, labels) = pickle.load(file)

    file.close()

    return pixels, labels

X, y = load_data()
X = preprocess_input(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

def build_model():
    resnet50_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    resnet50_model.trainable = False

    x = resnet50_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(10, activation='softmax')(x)

    model = Model(inputs=resnet50_model.input, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    return model

model = build_model()

aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.1,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         horizontal_flip=True,
                         brightness_range=[0.2,1.5],
                         fill_mode="nearest",
                         preprocessing_function=preprocess_input)

filepath="weights-{epoch:02d}-{val_accuracy:.2f}.keras"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reducelr_onplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
callbacks_list = [checkpoint, early_stop, reducelr_onplateau]

history = model.fit(aug.flow(X_train, y_train, batch_size=64),
                    epochs=10,
                    validation_data=aug.flow(X_test, y_test, batch_size=64),
                    callbacks=callbacks_list)

for layer in model.layers[-30:]:
    layer.trainable = True

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-5), metrics=['accuracy'])

history_fine_tune = model.fit(aug.flow(X_train, y_train, batch_size=64),
                    epochs=20,
                    validation_data=aug.flow(X_test, y_test, batch_size=64),
                    callbacks=callbacks_list)

model.save('resnet50model.keras')


def plot_model_history(model_history, acc='accuracy', val_acc='val_accuracy'):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(model_history.history[acc]) + 1), model_history.history[acc])
    axs[0].plot(range(1, len(model_history.history[val_acc]) + 1), model_history.history[val_acc])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history[acc]) + 1, max(1, len(model_history.history[acc]) // 10)))
    axs[0].legend(['train', 'val'], loc='best')

    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(
        np.arange(1, len(model_history.history['loss']) + 1, max(1, len(model_history.history['loss']) // 10)))
    axs[1].legend(['train', 'val'], loc='best')

    plt.savefig('training_history.png')

plot_model_history(history)
plot_model_history(history_fine_tune)