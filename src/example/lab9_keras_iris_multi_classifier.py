# lab9_keras_iris_multi_classifier

from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

dataFrame1 = read_csv("./data/iris_data/iris.data", header=None)
dataset = dataFrame1.values
# print(dataFrame1.head())
features = dataset[:, 0:4].astype(float)
labels = dataset[:, 4]
print(labels[:10])
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
print(type(encoded_Y), encoded_Y[:50])
print(encoded_Y[50:100])
print(encoded_Y[100:])

dummy_y = np_utils.to_categorical(encoded_Y)
print(type(dummy_y), dummy_y[:10])

def baseline_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=15, verbose=1)
kFold1 = KFold(n_splits=3, shuffle=True)
results = cross_val_score(estimator, features, dummy_y, cv=kFold1)
print("result=%s", results)
print("Acc:%.4f%%, std:(%.4f)"%(results.mean()*100, results.std()))