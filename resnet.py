import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras import backend, optimizers as K
from data_clean import NotMNIST
from keras.utils import plot_model
from keras.applications import ResNet50

img_width, img_height = 28,28
def make_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(198,198,3))

    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(512, activation='relu'))
    # add_model.add(Dropout(0.5))
    add_model.add(Dense(10, activation='softmax'))

    # Combine base model and my fully connected layers
    final_model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

    # Compile model
    final_model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    return final_model, final_model.summary()

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
            'notMNIST_small',
            target_size=(28,28),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            'notMNIST_small',
            target_size=(28,28),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical')

    model = make_model()
    model.fit_generator(
            train_generator,
            steps_per_epoch=2000 // batch_size,
            epochs=2,
            validation_data=validation_generator,
            validation_steps=800 // batch_size)

    model.save_weights('first_try.h5')

    score = model.evaluate_generator(validation_generator, 72 // batch_size + 1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
