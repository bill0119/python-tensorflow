import numpy
from keras.layers import Dense
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split


dataset1 = numpy.loadtxt('./data/diabete_data/pima-indians-diabetes.data',
                         delimiter=',')
print(type(dataset1), dataset1[0, :])
print(dataset1.size, dataset1.shape)
inputList = dataset1[:, 0:8]  # x1,x2....x8
resultList = dataset1[:, 8]  # y
print(inputList.shape, resultList.shape)

feature_train, feature_test, label_train, label_test =\
train_test_split(inputList, resultList, test_size=0.2,
                 random_state=123456)



model = load_model('.\\model\\save_keras1')

print(type(model.metrics_names))
print("metrics:", model.metrics_names)

scores = model.evaluate(feature_test, label_test)
print("\n\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))