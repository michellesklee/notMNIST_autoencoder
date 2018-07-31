import matplotlib as mpl
import os
import keras
import numpy as np
import gzip
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Flatten,Dropout
from keras.models import Model
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from cnn_autoencoder import extract_data, extract_labels

def encoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)

    return conv3

def plot_scatter(encode_func,test_data,test_label):
    X_test_repr = encode([test_data])[0]
    indices = random.sample(xrange(X_test_repr.shape[0]), 3000)
    points = X_test_repr[indices]
    labels = test_label[indices]

    norm = mpl.colors.Normalize(vmin=0, vmax=10)
    m = cm.ScalarMappable(norm=norm, cmap=cm.Paired)
    colors = m.to_rgba(labels)

    pl.figure(figsize=(12, 12))
    pl.title('Representation layer (2 neurons)')

    scatters = []
    for lid in np.unique(labels):
        mask = labels == lid
        sc = pl.scatter(points[mask, 0], points[mask, 1], c=colors[mask], linewidth=0.5, s=50)
        scatters.append(sc)
    pl.legend(scatters, np.arange(10), scatterpoints=1, loc='lower left', ncol=3, fontsize=12)

# For each label, plot an histogram of the encoding of
# all test example that belong to said label
def plot_hist(encode,func,test_data,test_label):
    X_test_repr = encode([test_data])[0]
    pl.figure(figsize=(18, 15))
    for label in np.unique(test_label):
        pl.subplot(3, 4, label + 1)
        pl.title(label)
        encodings = X_test_repr[test_label == label]

        # encodings is nexamples x 10
        means = np.mean(encodings, axis=0)
        stds = np.std(encodings, axis=0)

        bar_centers = np.arange(X_test_repr.shape[1])
        pl.bar(bar_centers, means, width=0.8, align='center', yerr=stds, alpha=0.5)
        pl.xticks(bar_centers, bar_centers)
        pl.xlim((-0.5, bar_centers[-1] + 0.5))

if __name__ == '__main__':
    train_data = extract_data('notMNIST_small.tar.gz', 14979)
    test_data = extract_data('notMNIST_small.tar.gz', 3745)
    train_labels = extract_labels('notMNIST_small.tar.gz', 14979)
    test_labels = extract_labels('notMNIST_small.tar.gz', 3745)

    encoder = Model(input_img,encoder(input_img))
    encode = K.function([encoder.get_input(train=False)], [encoder.get_output(train=False)])

    plot_hist(encode,test_data,test_labels))
    plt.show()
    plot_scatter(encode,test_data,test_labels)
