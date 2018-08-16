# lab8_keras_grid_cross_val_score
import numpy
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import StratifiedKFold

dataset1 = numpy.loadtxt('.\\data\\diabete_data\\pima-indians-diabetes.data',
                         delimiter=',')
inputList = dataset1[:, 0:8]  # x1,x2....x8
resultList = dataset1[:, 8]  # y


def create_default_model(optimizer='adam', init='uniform'):
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer=init
                    , activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model


model = KerasClassifier(build_fn=create_default_model, verbose=0)
optimizers = ['rmsprop', 'adam']
inits = ['normal', 'uniform']
epochs = [50, 100, 150]
batches = [5, 10, 15]
parameterGrid = dict(optimizer=optimizers, epochs=epochs,
                     batch_size=batches, init=inits)
grid = GridSearchCV(estimator=model, param_grid=parameterGrid)
grid_result = grid.fit(inputList, resultList)

print("Best:%f, using %s"%(grid_result.best_score_,
                           grid_result.best_params_))