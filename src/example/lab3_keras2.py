import numpy
from keras.layers import Dense
from keras.models import Sequential

numpy.random.seed(20180804)

dataset1 = numpy.loadtxt('.\\data\\diabete_data\\pima-indians-diabetes.data', delimiter=',')
print(type(dataset1), dataset1[0, :])
print(dataset1.size, dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
print(inputList.shape, resultList.shape)

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# use validation split
model.fit(inputList, resultList, validation_split=0.1, epochs=150, batch_size=10)
model.save('.\\model\\save_keras1')

print(type(model.metrics_names))
print("metrics:", model.metrics_names)

scores = model.evaluate(inputList, resultList)

print("\n\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))