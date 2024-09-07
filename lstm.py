import numpy as np
from Crypto.Cipher import ARC4
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Step 1: Dataset Generation
def generate_keystreams(num_streams, key_length):
    keystreams = []
    for _ in range(num_streams):
        key = np.random.bytes(key_length)
        cipher = ARC4.new(key)
        keystream = cipher.encrypt(b'\x00' * 256)
        keystreams.append([b for b in keystream])
    return np.array(keystreams)

# Generate and save the keystream dataset
keystreams = generate_keystreams(1000000, 16)
np.save('keystreams.npy', keystreams)

# Step 2: Data Preprocessing
# Load the dataset
keystreams = np.load('keystreams.npy')

# X should be the first 255 bytes, and y should be the 256th byte (target output)
X = keystreams[:, :-1].reshape(-1, 255, 1) / 256.0
y = to_categorical(keystreams[:, -1], num_classes=256)

# Split into training and validation sets
X_train, X_val = X[:800000], X[800000:]
y_train, y_val = y[:800000], y[800000:]

# Step 3: Model Design
# Define the LSTM model architecture
model = Sequential()

# Adding the first LSTM layer with 128 units and returning sequences
model.add(LSTM(128, input_shape=(255, 1), return_sequences=True))

# Adding a second LSTM layer with 128 units
model.add(LSTM(128, return_sequences=True))

# Adding a third LSTM layer with 128 units
model.add(LSTM(128))

# Output layer with 256 units (one for each possible byte value) and softmax activation
model.add(Dense(256, activation='softmax'))

# Compile the model with categorical crossentropy loss and Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 4: Training the Model
# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val))

# Save the trained model for later use
model.save('rc4_cracking_model.h5')

# Step 5: Model Evaluation
# Evaluate the model on the validation set
validation_loss, validation_accuracy = model.evaluate(X_val, y_val)

print(f'Validation Loss: {validation_loss}')
print(f'Validation Accuracy: {validation_accuracy}')
