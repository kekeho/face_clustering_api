import numpy as np
import face_recognition
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import random
from functools import reduce

from keras.layers import Input, Dense, Dropout
from keras.models import Model
import pickle
from keras.callbacks import TensorBoard
from keras import backend as K


matplotlib.use('pdf')


def _check_same_shape(images: np.array):
    first = images[0].shape
    for i in images[1:]:
        if i.shape != first:
            return False
    
    return True


def gen_autoencoder(input_dim, encoding_dim) -> Model:
    """
        モデル生成
    """
    input_img = Input(shape=(input_dim,))

    #  Encode部分
    encoded = Dense(64, activation='relu')(input_img)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    encoded = Dense(encoding_dim)(encoded)

    #  Decode部分
    decoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(32, activation="relu")(decoded)
    decoded = Dense(64, activation="relu")(decoded)
    decoded = Dense(128)(decoded)

    #  Model
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer="Adam", loss='mse')

    return autoencoder


def load_lfw(lfw_dir_path: os.PathLike) -> tuple:
    """Load lfw and return 128d encodings
    Arg:
        lfw_dir_path: lfw datasets path
    Returns:
        (x_train, x_test): (encodings[0:70%], encodings[70%:99%])
    """
    
    pickle_file = 'autoencoder_file_encodings.pkl'
    if os.path.exists(pickle_file):
        print("Using picle data")
        with open(pickle_file, 'rb') as f:
            encodings = pickle.load(f)
    else:
        filelist = glob(os.path.join(lfw_dir_path, '*/*[.jpg | .jpeg | .JPEG | .JPG | .png | .PNG]'))
        images = np.asarray([np.asarray(Image.open(f)) for f in filelist])

        # Calc face locations
        # if _check_same_shape(images):
        #     location_list = face_recognition.batch_face_locations(images)
        # else:
        #     location_list = [face_recognition.face_locations(i, model='cnn') for i in images]
        location_list = [face_recognition.face_locations(i, model='cnn') for i in images]

        # Calc face encodings (128d)
        encodings = []
        for img, loc in zip(images, location_list):
            encodings_per_image = face_recognition.face_encodings(img, loc)
            for enc in encodings_per_image:
                encodings.append(enc)
        with open(pickle_file, 'wb') as f:
            pickle.dump(encodings, f)

    per70 = int(len(encodings) * 0.7)

    return tuple(map(np.array, (encodings[:per70], encodings[per70:])))


def main():
    x_train, x_test = load_lfw('./images/lfw')

    # Normalize
    x_train += abs(x_train.min()) if x_train.min() < 0 else 0
    x_test += abs(x_test.min()) if x_test.min() < 0 else 0

    noise_factor = 0.01
    x_train_noised = x_train + noise_factor * np.random.normal(loc=0., scale=x_train.max(), size=x_train.shape)
    # x_train_noised = x_train  # NO NOISE

    autoencoder = gen_autoencoder(x_train.shape[1], 3)
    print(autoencoder)

    weights_file = 'autoencoder_weights.hdf5'
    if os.path.exists(weights_file):
        autoencoder.load_weights(weights_file)
    else:
        autoencoder.fit(x_train_noised, x_train, epochs=2000, batch_size=4098, shuffle=True, validation_data=(x_test, x_test), callbacks=[TensorBoard(log_dir='log_dir')])
        autoencoder.save_weights(weights_file)

    x_test_noised = x_test + noise_factor * np.random.normal(loc=0., scale=x_train.max(), size=x_test.shape)
    # decoded_array = autoencoder.predict(x_test_noised)

    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[4].output)
    encoded = encoder.predict(x_test_noised)

    
    encoded_input = Input(shape=(3,))
    decoder_layer = autoencoder.layers[5](encoded_input)
    for l in autoencoder.layers[6:]:
        decoder_layer = l(decoder_layer)
    decoder = Model(encoded_input, decoder_layer)
    decoded_array = decoder.predict(encoded)


    n = 5
    image_test = x_test.reshape(x_test.shape[0], 8, 16)
    image_test_noised = x_test_noised.reshape(x_test_noised.shape[0], 8, 16)
    image_encoded = encoded.reshape(encoded.shape[0], 1, 3)
    image_decoded = decoded_array.reshape(decoded_array.shape[0], 8, 16)

    plt.figure(figsize=(20, 10))
    for i in range(n):
        index = random.randint(0, x_test.shape[0]-1)
        plt.subplot(4, n, i+1)
        plt.imshow(image_test[index])

        plt.subplot(4, n, i+1+n)
        plt.imshow(image_test_noised[index])
        
        plt.subplot(4, n, i+1+2*n)
        plt.imshow(image_encoded[index])

        plt.subplot(4, n, i+1+3*n)
        plt.imshow(image_decoded[index])

    print(encoded[:n])
    
    plt.savefig('hoge.pdf')



if __name__ == "__main__":
    main()



