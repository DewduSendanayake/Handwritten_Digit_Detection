import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 1) Load & reshape data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1,28,28,1) / 255.0
x_test  = x_test.reshape(-1,28,28,1) / 255.0

# 2) Set up augmentation: small rotations & shifts
aug = ImageDataGenerator(
    rotation_range=10,       # rotate by up to 10°
    width_shift_range=0.1,   # horizontal shift up to 10% of width
    height_shift_range=0.1   # vertical shift up to 10% of height
)

# 3) Build CNN with Dropout
model = models.Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),               # drop 25% of conv outputs
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),                # drop 50% before final layer
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 4) Train with augmented data generator
print("Training with augmentation—this might take a bit longer...")
history = model.fit(
    aug.flow(x_train, y_train, batch_size=64),
    epochs=5,
    validation_data=(x_test, y_test)
)

# 5) Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nAugmented CNN Test accuracy: {test_acc:.4f}")

# 6) Plot accuracy curves
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 7) Save the improved model
model.save('mnist_cnn_augmented.h5')
print("Saved as mnist_cnn_augmented.h5")
