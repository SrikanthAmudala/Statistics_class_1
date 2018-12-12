'''
@auther: Srikanth Amudala | Deeksha
@desc: PCA vs Autoencoder using mnist handwritten digits dataset
'''

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense


def plot_train_history_loss(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    # plt.show()


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784) / 255
x_test = x_test.reshape(10000, 784) / 255

# Principal Component Analysis
mu = x_train.mean(axis=0)
U, s, V = np.linalg.svd(x_train - mu, full_matrices=False)
Zpca = np.dot(x_train - mu, V.transpose())
Rpca = np.dot(Zpca[:, :2], V[:2, :]) + mu  # reconstruction
err = np.sum((x_train - Rpca) ** 2) / Rpca.shape[0] / Rpca.shape[1]
print('PCA reconstruction error with 2 PCs: ' + str(round(err, 3)))

# Autoencoder
model = Sequential([
    Dense(512, activation='relu', input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(2, activation='relu', name="code"),
    Dense(128, activation='relu'),
    Dense(512, activation='relu'),
    Dense(784, activation='sigmoid')
])
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(x_train, x_train, batch_size=128, epochs=5, verbose=1,
                    validation_data=(x_test, x_test))

plot_train_history_loss(history)
# plt.show()

encoder = Model(model.input, model.get_layer('code').output)
Zenc = encoder.predict(x_train)  # code representation
Renc = model.predict(x_train)  # reconstruction

# Plotting PCA projection side-by-side with the bottleneck representation
plt.figure(figsize=(8, 4), num=2)
plt.subplot(1, 2, 1)
plt.title('PCA')
plt.scatter(Zpca[:5000, 0], Zpca[:5000, 1], c=y_train[:5000], s=8, cmap='tab10')
plt.gca().get_xaxis().set_ticklabels([])
plt.gca().get_yaxis().set_ticklabels([])

plt.subplot(1, 2, 2)
plt.title('Autoencoder')
plt.scatter(Zenc[:5000, 0], Zenc[:5000, 1], c=y_train[:5000], s=8, cmap='tab10')
plt.gca().get_xaxis().set_ticklabels([])
plt.gca().get_yaxis().set_ticklabels([])



# plt.show()

# Reconstructions (1st row - original images, 2nd row - PCA, 3rd row - autoencoder):
plt.figure(figsize=(9, 3), num=3)
toPlot = (x_train, Rpca, Renc)
for i in range(10):
    for j in range(3):
        ax = plt.subplot(3, 10, 10 * j + i + 1)
        plt.imshow(toPlot[j][i, :].reshape(28, 28), interpolation="nearest",
                   vmin=0, vmax=1)
        plt.gray()
        # plt.xlabel()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()

