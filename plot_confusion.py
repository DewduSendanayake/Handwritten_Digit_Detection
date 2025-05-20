import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1) Load the saved augmented model
model = tf.keras.models.load_model('mnist_cnn_augmented.h5')

# 2) Load & preprocess test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.reshape(-1,28,28,1) / 255.0

# 3) Predict labels for the test set
y_prob = model.predict(x_test)
y_pred = np.argmax(y_prob, axis=1)

# 4) Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# 5) Plot it!
disp = ConfusionMatrixDisplay(cm, display_labels=range(10))
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('MNIST Confusion Matrix')
plt.show()
