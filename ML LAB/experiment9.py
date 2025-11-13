
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

# Step 1: Load and prepare the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Step 2: Build the Model
model = Sequential([
    Flatten(input_shape=(28, 28)),      # Flatten 2D image to 1D vector
    Dense(128, activation='relu'),      # Hidden layer with ReLU
    Dense(10, activation='softmax')     # Output layer (10 digits)
])

# Step 3: Compile the Model
model.compile(
    optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(),
    metrics=[SparseCategoricalAccuracy()]
)

# Step 4: Train the Model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Step 5: Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")


