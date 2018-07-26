import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras import backend as K
from data_clean import NotMNIST
from keras.utils import plot_model

img_width, img_height = 28,28
def make_model():
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 1)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    np.random.seed(13)

    batch_size = 16

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=(28,28),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            'data/validation',
            target_size=(28,28),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical')

    model = make_model()
    model.fit_generator(
            train_generator,
            steps_per_epoch=2000 // batch_size,
            epochs=20,
            validation_data=validation_generator,
            validation_steps=800 // batch_size)

    model.save_weights('first_try.h5')

    score = model.evaluate_generator(validation_generator, 72 // batch_size + 1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    plot_model(model, show_shapes=True, to_file = 'model.png')
