import tensorflow as tf
import numpy as np

(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.mnist.load_data()
print('load images ready')
flattenDim = 28 * 28
TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)
trainImages = np.reshape(train_images, (TRAINING_SIZE, flattenDim))
testImages = np.reshape(test_images, (TEST_SIZE, flattenDim))
# transfer to floating
trainImages = trainImages.astype(np.float32)
testImages = testImages.astype(np.float32)
print(trainImages[0])
# convert to range0-1
trainImages /= 255
testImages /= 255
print(trainImages[0])
# convert 1-hot encoding
NUM_DIGITS = 10
trainLabels = tf.keras.utils.to_categorical(train_labels, NUM_DIGITS)
testLabels = tf.keras.utils.to_categorical(test_labels, NUM_DIGITS)
# create keras model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu,
                                input_shape=(flattenDim,)))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print(model.summary())
#train model
model.fit(trainImages, trainLabels, epochs=10)
loss, accuracy = model.evaluate(testImages, testLabels)
print('tets accuracy:%.4f'%(accuracy))