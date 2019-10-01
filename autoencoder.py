import numpy as np
import face_recognition
import os
from glob import glob
from PIL import Image

from keras.layers import Input, Dense
from keras.models import Model
import pickle
from keras.callbacks import TensorBoard


def _check_same_shape(images: np.array):
    first = images[0].shape
    for i in images[1:]:
        if i.shape != first:
            return False
    
    return True


def gen_autoencoder(input_dim, encoding_dim) -> Model:
    """Generate simple autoencoder
    Args:
        input_dim: input shape
        encoding_dim: encoded shape
    
    Returns:
        (autoencoder, encoder)
    """
    input_face_encoding = Input(shape=(input_dim,))

    # Encode
    encoded = Dense(encoding_dim, activation='relu')(input_face_encoding)
    encoder = Model(input_face_encoding, encoded)

    # Decode
    decoded = Dense(input_dim, activation='relu')(encoded)

    autoencoder = Model(input_face_encoding, decoded)

    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    # Compile
    autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

    return (autoencoder, encoder, decoder)


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
    autoencoder, encoder, decoder = gen_autoencoder(x_train.shape[1], 3)

    autoencoder.fit(x_train, x_train, epochs=1000, batch_size=1024, shuffle=True, validation_data=(x_test, x_test), callbacks=[TensorBoard(log_dir='log_dir')])

    encoded_array = encoder.predict(x_test)
    print(encoded_array.shape)
    decoded_array = decoder.predict(encoded_array)

    for test, decoded in zip(x_test[:5], decoded_array[5:]):
        print(decoded - test)


if __name__ == "__main__":
    main()



