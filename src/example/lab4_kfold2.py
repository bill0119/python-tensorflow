# lab4_keras_kfold
# use k-fold to resample data
import numpy
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

numpy.random.seed(20180811)

dataset1 = numpy.loadtxt('.\\data\\diabete_data\\pima-indians-diabetes.data', delimiter=',')
print(type(dataset1), dataset1[0, :])
print(dataset1.size, dataset1.shape)
inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
print(inputList.shape, resultList.shape)

fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
TotalScore = []

for train, test in fiveFold.split(inputList, resultList):
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(inputList[train], resultList[train], epochs=150, batch_size=10, verbose=0)

    print(type(model.metrics_names))
    print("metrics:", model.metrics_names)

    scores = model.evaluate(inputList[test], resultList[test])
    TotalScore.append(scores[0]*100)
    print("get a price of result:%.3f" % (score[1] * 100))

# print("\n\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
