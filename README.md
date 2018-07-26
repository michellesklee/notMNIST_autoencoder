![sampleletters](https://github.com/michellesklee/notMNIST_autoencoder/blob/master/images/sample_letters.png)

## Definitely *not* MNIST

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is so 1998. In 2011 Yaroslav Bulatov made the [notMNIST dataset](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html). There are still 10 classes, but instead of numbers 0 - 9 you are classifying letters A - J of many different font styles.

## Model Building
Benchmark: Yaroslav Bulatov did logistic regression on top of autoencoding and got 84% accuracy.

#### Model 1
Basic autoencoder: 16.77% accuracy

![first_pass](https://github.com/michellesklee/notMNIST_autoencoder/blob/master/images/first_pass.png)

```python
def autoencoder_model(X_train):
    input_img = Input(shape=(X_train.shape[1],))
    encoded1 = Dense(units = 256, activation = 'relu')(input_img)
    encoded2 = Dense(units = 64, activation='relu')(encoded1)
    decoded1 = Dense(units = 256, activation='relu')(encoded2)
    decoded2 = Dense(units = 784, activation='sigmoid')(decoded1)
    autoencoder = Model(input_img, decoded2)
    autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mse'])

    return autoencoder
```

#### Model 2
Multilayer autoencoder: 27.51% accuracy

Layer | Output Shape | Param #
--- | --- | ---
input_2 (InputLayer)     |    (None, 28, 28, 1)    |     0
conv2d_1 (Conv2D)       |     (None, 28, 28, 32)     |   320
max_pooling2d_1 (MaxPooling2)| (None, 14, 14, 32)     |   0
conv2d_2 (Conv2D)     |       (None, 14, 14, 64)  |      18496
max_pooling2d_2 (MaxPooling2)| (None, 7, 7, 64)       |   0
conv2d_3 (Conv2D)    |        (None, 7, 7, 128)     |    73856
conv2d_4 (Conv2D)     |       (None, 7, 7, 128)      |   147584
up_sampling2d_1 (UpSampling2)| (None, 14, 14, 128)    |   0
conv2d_5 (Conv2D)            |(None, 14, 14, 64)       | 73792
up_sampling2d_2 (UpSampling2)| (None, 28, 28, 64)       | 0
conv2d_6 (Conv2D)           | (None, 28, 28, 1)        | 577

**Total params** 314,625

```python
def autoencoder(input_img):
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    up1 = UpSampling2D((2,2))(conv4)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    up2 = UpSampling2D((2,2))(conv5)
    decoded = Conv2D(1, (3, 3), activation='softmax', padding='same')(up2)

    return decoded

autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss = 'sparse_categorical_crossentropy', metrics=['accuracy'], optimizer = 'adam')
```

#### Model 3: CNN
CNN: 76.25% accuracy   

![model](https://github.com/michellesklee/notMNIST_autoencoder/blob/master/images/model.png)

```python
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
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
 ```
#### CNN on Large Dataset 
Test-train split and running our CNN model resulted in 62.5% accuracy

#### Future Steps
While we were not able to reach at least 84% accuracy, there are a number of future steps we could take:

1. Continue to fine tune the CNN model
2. Try pre-trained model (e.g., ResNet50)

