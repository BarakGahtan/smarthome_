import tensorflow as tf
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense, Reshape
from tensorflow import keras
from tensorflow.keras import layers
TIME_PERIODS  = 319

model_m = Sequential()
model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=((4,1024,319),)))
model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(num_classes, activation='softmax')) #fully connected
print(model_m.summary())

￿