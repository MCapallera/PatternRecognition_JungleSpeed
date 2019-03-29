from __future__ import print_function
import random
import matplotlib.pyplot as plt
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from util.dataset import get_training_set, get_test_set


batch_size = 128
num_classes = 10
#epochs = 10

# the data, split between train and test sets

y_train, x_train = get_training_set()
y_test, x_test = get_test_set()


print(len(x_train))
print(len(x_test))


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

lr_list = [0.05, 0.075, 0.1]
nb_epochs_list = [35, 40, 45, 50, 55, 60]

#for k in range(0,11):
    # nb_neurons = random.randint(50, 80)
    # lr = random.choice(lr_list)
    # epochs = random.choice(nb_epochs_list)

nb_neurons = 62
lr = 0.1
epochs = 40

model = Sequential()
model.add(Dense(nb_neurons, activation='relu', input_shape=(784,)))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

sgd = optimizers.SGD(lr=lr)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print("For " + str(epochs) + " epochs; lr = " + str(lr) + "; number of neurones = " + str(nb_neurons))
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'])
plt.show()