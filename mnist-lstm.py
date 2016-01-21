
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.initializations import normal, identity, one
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.optimizers import RMSprop
from keras.utils import np_utils
import sys, os
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

nb_classes = 10
nb_epochs = 1000

batch_size = int(sys.argv[1]) # 100
hidden_units = int(sys.argv[2]) # 50

# rmsprop
learning_rate = float(sys.argv[3]) # 0.001
rho = float(sys.argv[4]) # 0.9

clip_norm = float(sys.argv[5]) #5.0
forget_bias = 1.0

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1, 1)
X_test = X_test.reshape(X_test.shape[0], -1, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_means = np.mean(X_train, axis=0)
X_stds = np.std(X_train, axis=0)
X_train = (X_train - X_means)/(X_stds+1e-6)
X_test = (X_test - X_means)/(X_stds+1e-6)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# lambda shape: forget_bias*one(shape, name=None)

print('Compare to LSTM...')
model = Sequential()
model.add(LSTM(hidden_units, input_shape=X_train.shape[1:], inner_init='glorot_uniform',
forget_bias_init='one', activation='tanh', inner_activation='sigmoid'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop( clipnorm=clip_norm)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop)


checkpointer = ModelCheckpoint(filepath="/scratch/pponte/data/rnn-mist/lstm-weights"+"-bs-"+str(batch_size)+"-hu-"+str(hidden_units)+"-lr-"+str(learning_rate)+"-rho-"+str(rho)+"-clip-"+str(clip_norm)+"-epoch-{epoch:02d}-val-{val_acc:.2f}"+".hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_acc', patience=15, verbose=1, mode='max')
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test),
          callbacks=[earlystopper, checkpointer]
          )

scores = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('LSTM test score:', scores[0])
print('LSTM test accuracy:', scores[1])
