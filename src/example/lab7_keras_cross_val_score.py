# lab7_cross_val_score

import numpy
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import StratifiedKFold

dataset1 = numpy.loadtxt('./data/diabete_data/pima-indians-diabetes.data',
                         delimiter=',')
inputList = dataset1[:, 0:8]  # x1,x2....x8
resultList = dataset1[:, 8]  # y


def create_default_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


model = KerasClassifier(build_fn=create_default_model,
                        epochs=150, batch_size=20, verbose=0)
fiveFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
results = cross_val_score(model, inputList, resultList, cv=fiveFold)
print("result=", results)
