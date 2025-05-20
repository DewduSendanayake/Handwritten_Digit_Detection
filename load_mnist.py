import tensorflow as tf

# 1) Download & split the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2) Print out the shapes to verify everything loaded correctly
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test  shape:", x_test.shape)
print("y_test  shape:", y_test.shape)
