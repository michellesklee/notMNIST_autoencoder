![sampleletters](https://github.com/michellesklee/notMNIST_autoencoder/blob/master/images/sample_letters.png)

## Definitely *not* MNIST

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is so 1998. In 2011 Yaroslav Bulatov made the [notMNIST dataset](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html). There are still 10 classes, but instead of numbers 0 - 9 you are classifying letters A - J of many different font styles.

## Data Cleaning

## Model Building
Yaroslav Bulatov did logistic regression on top of autoencoding and got 84% accuracy - this was our benchmark.

#### Model 1
Basic autoencoder: 16.77% accuracy

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
CNN: 76.25% accuracy

![model](https://github.com/michellesklee/notMNIST_autoencoder/blob/master/images/model.png)
