import os
import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Flatten,Dropout
from keras.models import Model
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.utils import plot_model


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, 28,28)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels

label_dict = {
 0: 'A',
 1: 'B',
 2: 'C',
 3: 'D',
 4: 'E',
 5: 'F',
 6: 'G',
 7: 'H',
 8: 'I',
 9: 'J',
}

train_data = extract_data('notMNIST_small.tar.gz', 14979)
test_data = extract_data('notMNIST_small.tar.gz', 3745)
train_labels = extract_labels('notMNIST_small.tar.gz', 14979)
test_labels = extract_labels('notMNIST_small.tar.gz', 3745)

# plt.figure(figsize=[5,5])
#
# # Display the first image in training data
# plt.subplot(121)
# curr_img = np.reshape(train_data[1], (28,28))
# curr_lbl = train_labels[1]
# plt.imshow(curr_img, cmap='gray')
# #plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
#
# # Display the first image in testing data
# plt.subplot(122)
# curr_img = np.reshape(test_data[1], (28,28))
# curr_lbl = test_labels[1]
# plt.imshow(curr_img, cmap='gray')
# #plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
# plt.show()

train_data = train_data.reshape(-1, 28,28, 1)
test_data = test_data.reshape(-1, 28,28, 1)
train_data.shape, test_data.shape

train_data = train_data / np.max(train_data)
test_data = test_data / np.max(test_data)

train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,
                                                             train_data,
                                                             test_size=0.2,
                                                             random_state=13)

batch_size = 128
epochs = 50
inChannel = 1
x, y = 28, 28
input_img = Input(shape = (x, y, inChannel))

def autoencoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    #decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='softmax', padding='same')(up2) # 28 x 28 x 1
    return decoded

autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss = 'sparse_categorical_crossentropy', metrics=['accuracy'], optimizer = 'adam')
plot_model(autoencoder, show_shapes=True, to_file = 'model.png')

#autoencoder.summary() #see summary table in README

autoencoder.fit(train_data, train_data, epochs=10, batch_size=100, verbose=1, validation_split=0.1)
scores = autoencoder.evaluate(test_data, test_data)
print("Test accuracy = {}".format(scores[0]))
