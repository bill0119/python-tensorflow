# lab12_load_mnist
import tensorflow as tf
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

print('train image shape=%s, testimage shape=%s'%(train_images.shape, test_images.shape))
print('train label len=%d, test label len=%d'%(len(train_labels), len(test_labels)))

def plotImage(index):
    plt.title("The image marked as %d"%train_labels[index])
    plt.imshow(train_images[index], cmap='binary')
    plt.show()

def plotTestImage(index):
    plt.title("The image marked as %d"%test_labels[index])
    plt.imshow(test_images[index], cmap='binary')
    plt.show()

plotImage(200)
plotTestImage(200)

