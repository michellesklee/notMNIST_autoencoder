![sampleletters](https://github.com/michellesklee/notMNIST_autoencoder/blob/master/images/sample_letters.png)

## Definitely *not* MNIST

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is so 1998. In 2011 Yaroslav Bulatov made the [notMNIST dataset](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html). There are still 10 classes, but instead of numbers 0 - 9 you are classifying letters A - J of many different font styles.

## Model Building
Yaroslav Bulatov did logistic regression on top of autoencoding and got 84% accuracy - this was our benchmark.

#### Model 1
Basic autoencoder: 16.77% accuracy

```python
def autoencoder_model(X_train):
    '''
    defines autoencoder model
    input: X_train (2D np array)
    output: autoencoder (compiled autoencoder model)
    '''
    input_img = Input(shape=(X_train.shape[1],))
    encoded1 = Dense(units = 256, activation = 'relu')(input_img)
    encoded2 = Dense(units = 64, activation='relu')(encoded1)
    decoded1 = Dense(units = 256, activation='relu')(encoded2)
    decoded2 = Dense(units = 784, activation='sigmoid')(decoded1)
    autoencoder = Model(input_img, decoded2)
    autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

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

#### Model 3: CNN
CNN: 73.75% accuracy   

![model](https://github.com/michellesklee/notMNIST_autoencoder/blob/master/images/model.png)
