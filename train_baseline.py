import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1) Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2) Normalize pixels to [0,1]
x_train = x_train / 255.0
x_test  = x_test  / 255.0

# 3) Flatten images: 28×28 → 784
x_train_flat = x_train.reshape(-1, 28*28)
x_test_flat  = x_test.reshape(-1, 28*28)

# 4) Train a Logistic Regression
print("Training Logistic Regression (this may take a minute)...")
lr = LogisticRegression(
    multi_class='multinomial',
    solver='saga',
    max_iter=100,
    n_jobs=-1     # use all your CPU cores
)
lr.fit(x_train_flat, y_train)

# 5) Evaluate
y_pred = lr.predict(x_test_flat)
acc = accuracy_score(y_test, y_pred)
print(f"Baseline Logistic Regression Accuracy: {acc:.4f}")
