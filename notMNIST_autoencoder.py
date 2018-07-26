from data_clean import NotMNIST
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras import regularizers
import matplotlib.pyplot as plt

mnist = NotMNIST()

def autoencoder_model(X_train):
    '''
    defines autoencoder model
    input: X_train (2D np array)
    output: autoencoder (compiled autoencoder model)
    '''
    # this is our input placeholder
    input_img = Input(shape=(X_train.shape[1],))

    # first encoding layer
    encoded1 = Dense(units = 256, activation = 'relu')(input_img)

    # second encoding layer
    # note that each layer is multiplied by the layer before
    encoded2 = Dense(units = 64, activation='relu')(encoded1)

    # first decoding layer
    decoded1 = Dense(units = 256, activation='relu')(encoded2)

    # second decoding layer - this produces the output
    decoded2 = Dense(units = 784, activation='sigmoid')(decoded1)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded2)

    # compile model
    autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

    return autoencoder

def plot_reconstruction(X_orig, X_decoded, n = 10, plotname = None):
    '''
    inputs: X_orig (2D np array of shape (nrows, 784))
            X_recon (2D np array of shape (nrows, 784))
            n (int, number of images to plot)
            plotname (str, path to save file)
    '''
    plt.figure(figsize=(n*2, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_orig[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(X_decoded[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if plotname:
        plt.savefig(plotname)
    else:
        plt.show()

if __name__ == '__main__':
    X_train = np.asarray(mnist.train.images)
    X_test = np.asarray(mnist.test.images)
    np.random.seed(42)
    model = autoencoder_model(X_train)
    batch_size = 100
    model.fit(X_train, X_train, epochs=100, batch_size=batch_size, verbose=1,
              validation_split=0.1)
    scores = model.evaluate(X_test, X_test)
    print("Test accuracy = {}".format(scores[0]))

    X_test_decoded = model.predict(X_test)

    plot_reconstruction(X_test, X_test_decoded, plotname='images/first_pass.png') #accuracy = 16%
