import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical( y_test)

model = tf.keras.models.Sequential([
   tf.keras.layers.Conv2D(16, (3, 3), activation='relu',padding='same', input_shape=(28, 28,1)),
   tf.keras.layers.MaxPooling2D((2, 2)),
   tf.keras.layers.Conv2D(32, (5, 5), activation='relu',padding='same'),
   tf.keras.layers.MaxPooling2D((2, 2)),
   tf.keras.layers.Flatten(),
   tf.keras.layers.Dense(256, activation='relu'),
   tf.keras.layers.Dropout(0.5),
   tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=1)
model.evaluate(x_test, y_test, verbose=2)

tf.keras.models.save_model(model,'asd.hdf5')