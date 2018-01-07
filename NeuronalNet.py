from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, ZeroPadding2D
from keras.models import Model
from keras import losses
from keras.utils import plot_model

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
import keras

test_data_size = 0.2
validation_data_size = 0.2
epoches = 2

mapping={'0': [0,0,0,0,0,0],
         '1': [0, 0, 0, 0, 0, 1],
         '2': [0, 0, 0, 0, 1, 0],
         '3': [0, 0, 0, 0, 1, 1],
         '4': [0, 0, 0, 1, 0, 0],
         '5': [0, 0, 0, 0, 0, 1],
         '6': [0, 0, 0, 1, 1, 0],
         '7': [0, 0, 0, 1, 1, 1],
         '8': [0, 0, 1, 0, 0, 0],
         '9': [0, 0, 1, 0, 0, 1]}

print('Load data')
data = pd.read_csv('data.csv')

cut = int(len(data) * (1 - test_data_size))
train_data = data[:cut]
test_data = data[cut:]

cut = int(len(train_data) * (1 - validation_data_size))
validation_data = train_data[cut:]
train_data = train_data[:cut]

x_train_symbolType = np.array(train_data['SymbolType'])
x_train_layout = np.array([mapping[str(i)] for i in train_data['Layout']])
x_train_Circuit = np.array([mapping[str(i)] for i in train_data['CircuitDiagram']])
y_train = np.array([mapping[str(i)] for i in train_data['Result']])

x_test_symbolType = np.array(test_data['SymbolType'])
x_test_layout = np.array([mapping[str(i)] for i in test_data['Layout']])
x_test_Circuit = np.array([mapping[str(i)] for i in test_data['CircuitDiagram']])
y_test = np.array([mapping[str(i)] for i in test_data['Result']])

x_validation_symbolType = np.array(validation_data['SymbolType'])
x_validation_layout = np.array([mapping[str(i)] for i in validation_data['Layout']])
x_validation_Circuit = np.array([mapping[str(i)] for i in validation_data['CircuitDiagram']])
y_validation = np.array([mapping[str(i)] for i in validation_data['Result']])

x_fit_train = {'x_layout': x_train_layout, 'x_circuit': x_train_Circuit, 'x_symbolType' : x_train_symbolType}
x_fit_validation = {'x_layout': x_validation_layout, 'x_circuit': x_validation_Circuit, 'x_symbolType' : x_validation_symbolType}
x_fit_test = {'x_layout': x_test_layout, 'x_circuit': x_test_Circuit, 'x_symbolType' : x_test_symbolType}

print('Build model')

inputLayout = Input(shape=(6,), name="x_layout")
inputCircuit = Input(shape=(6,), name="x_circuit")
inputSymbolType = Input(shape=(1,), dtype='float32', name="x_symbolType")

layers = keras.layers.concatenate([inputSymbolType, inputCircuit, inputLayout])
layers = Dense(13, activation='relu')(layers)
layers = Dense(6, activation='softmax')(layers)

model = Model(inputs=[inputLayout, inputCircuit, inputSymbolType], outputs=layers)

opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss=losses.mean_squared_logarithmic_error,
              optimizer=opt,
              metrics=['accuracy'])

print('Train model')
model.fit(x_fit_train, y=y_train,
                         epochs=epoches,
                         shuffle=True,
                         validation_data=(x_fit_validation, y_validation))

print('Evaluate model')
scores = model.evaluate(x_fit_test, y_test, verbose=1)
print(scores)

print('plot model')
plot_model(model=model, to_file='model.png', show_shapes=True, show_layer_names=True)

print('plot confusion_matrix')

cls_pred = model.predict(x=x_fit_test)
cls_pred = [label.argmax() for label in cls_pred]
y_test_cls = [label.argmax() for label in y_test]
cm = confusion_matrix(y_true=y_test_cls,
                      y_pred=cls_pred)

# normalize Data
summedValues = np.sum(cm, axis=1)
summedValues = np.reshape(summedValues, [-1, 1])
print(cm)
cm_normalized = np.divide(cm, summedValues)

sn.heatmap(cm_normalized, annot=True, linewidths=.2, cmap="YlGnBu")
num_actions = len(y_test[0])
tick_marks = np.arange(num_actions)
plt.xticks(tick_marks, range(num_actions))
plt.yticks(tick_marks, range(num_actions))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig("confusion_matrix.png")
plt.close()