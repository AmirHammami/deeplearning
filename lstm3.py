import numpy as np
from Crypto.Cipher import ARC4
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, BatchNormalization, Attention, Input, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
import tensorflow as tf

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
# Define the BiLSTM model with attention and residual connections

def attention_block(inputs):
    attention = Attention(use_scale=True)([inputs, inputs])
    return attention

# Sequential model with Bidirectional LSTM and Attention
def build_model():
    inputs = Input(shape=(255, 1))
    
    # Bidirectional LSTM
    lstm1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputs)
    batchnorm1 = BatchNormalization()(lstm1)
    
    # Attention Layer
    attention = attention_block(batchnorm1)
    
    # Add residual connection
    residual = Add()([attention, batchnorm1])
    
    # Second Bidirectional LSTM Layer
    lstm2 = Bidirectional(LSTM(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))(residual)
    
    # Output layer with 256 units (one for each possible byte value) and softmax activation
    outputs = Dense(256, activation='softmax')(lstm2)

    # Define model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = build_model()

# Step 4: Learning Rate Scheduler and Callbacks
def lr_schedule(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 10
    return initial_lr * (drop ** np.floor((1 + epoch) / epochs_drop))

lr_scheduler = LearningRateScheduler(lr_schedule)

# Early stopping and checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_rc4_model_bilstm_attention.h5', monitor='val_loss', save_best_only=True)

# Compile the model
optimizer = Adam(learning_rate=0.001, clipvalue=1.0)  # Gradient clipping
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Training the Model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, 
                    validation_data=(X_val, y_val), 
                    callbacks=[early_stopping, checkpoint, lr_scheduler])

# Save the trained model
model.save('rc4_bilstm_attention_model.h5')

# Step 6: Model Evaluation
# Evaluate the model on the validation set
validation_loss, validation_accuracy = model.evaluate(X_val, y_val)

print(f'Validation Loss: {validation_loss}')
print(f'Validation Accuracy: {validation_accuracy}')

# Step 7: Plot Training History
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
