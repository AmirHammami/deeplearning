import numpy as np
from Crypto.Cipher import ARC4
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
from tensorflow.keras.mixed_precision import set_global_policy
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Enable mixed precision for faster training
set_global_policy('mixed_float16')

# Multi-GPU strategy
import tensorflow as tf
strategy = tf.distribute.MirroredStrategy()

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

# Step 3: Model Design with strategy scope for multi-GPU
with strategy.scope():
    # Define the LSTM model architecture
    model = Sequential()

    # Adding the first LSTM layer with 128 units and Dropout
    model.add(LSTM(128, input_shape=(255, 1), return_sequences=True, dropout=0.2, recurrent_dropout=0.2))

    # Adding a second LSTM layer with 128 units and Dropout
    model.add(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))

    # Adding a third LSTM layer with 128 units and Dropout
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

    # Output layer with 256 units (one for each possible byte value) and softmax activation
    model.add(Dense(256, activation='softmax'))

    # Learning rate schedule function
    def lr_schedule(epoch):
        initial_lr = 0.001
        drop = 0.5
        epochs_drop = 10
        return initial_lr * (drop ** np.floor((1 + epoch) / epochs_drop))

    # Compile the model with categorical crossentropy loss, Adam optimizer, and additional metrics
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])

# Step 4: Callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=256, 
                    validation_data=(X_val, y_val), 
                    callbacks=[early_stopping, checkpoint, lr_scheduler])

# Save the trained model
model.save('rc4_cracking_model.h5')

# Step 5: Model Evaluation
# Evaluate the model on the validation set
validation_loss, validation_accuracy, precision, recall = model.evaluate(X_val, y_val)

print(f'Validation Loss: {validation_loss}')
print(f'Validation Accuracy: {validation_accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

# Step 6: Plot Training History
# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
