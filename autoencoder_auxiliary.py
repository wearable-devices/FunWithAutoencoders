from keras.layers import Input, Dense
from keras.layers import Lambda
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras import activations
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.signal as signal
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import Regularizer
import keras.backend as K

sequence_length = 8 # 32

from keras.layers import LeakyReLU, ReLU


def hard_sigmoid_approximation(x, threshold=0.5):
    """
    Approximates the Heaviside step function using keras.activations.hard_sigmoid.
    Args:
    - x: Input tensor.
    - threshold: Threshold value to determine the step. Values above this threshold are set to 1, below are set to 0.
    Returns:
    - A tensor where each element is either 0 or 1, approximating the Heaviside step function.
    """
    hard_sigmoid_output = K.hard_sigmoid(x)  # Use K.sigmoid instead of abs(K.sigmoid(x)) to get the correct hard sigmoid behavior
    # Cast the boolean tensor to float before multiplication
    step_function = K.cast(hard_sigmoid_output >= threshold, K.floatx())
    return K.cast(step_function, K.floatx())

# Standardization function to be used in Lambda layer
def standardize_input(x):
    return (x - K.mean(x)) / (K.std(x) + K.epsilon())

def create_autoencoder_and_extract_models(sequence_length, learning_rate=0.001):
    # Encoder definition
    input_layer = Input(shape=(sequence_length,))
    # standardization_layer = Lambda(standardize_input)(input_layer)
    encoder_hidden_1 = Dense(sequence_length // 2, activation='tanh')(input_layer)
    encoder_hidden_2 = Dense(sequence_length // 4, activation='tanh')(encoder_hidden_1)  # gelu
    latent = Dense(2, activation='tanh')(encoder_hidden_2)  # selu, elu, swish, gelu
    # latent = LeakyReLU(alpha=0.1)(latent)
    # latent = x = ReLU(max_value=6)(latent)
    encoder = Model(input_layer, latent)

    # Decoder definition
    encoded_input = Input(shape=(2,))
    decoder_hidden_1 = Dense(sequence_length // 4, activation='tanh')(encoded_input) # gelu
    decoder_hidden_2 = Dense(sequence_length // 2, activation='tanh')(decoder_hidden_1) # gelu
    # decoder_hidden_3 = Dense(sequence_length // 2, activation='gelu')(decoder_hidden_2)
    # decoder_hidden_4 = Dense(sequence_length, activation='gelu')(decoder_hidden_3)
    output = Dense(sequence_length, activation='tanh')(decoder_hidden_2)
    decoder = Model(encoded_input, output)

    # Autoencoder definition
    autoencoder_model = Model(input_layer, decoder(encoder(input_layer)))
    adam_optimizer = Adam(learning_rate=learning_rate)
    autoencoder_model.compile(optimizer=adam_optimizer, loss='mean_squared_error')  # mean_squared_error

    return autoencoder_model, encoder, decoder


# from functools import partial
#
# learning_rate = 0.001
# binary_reg_strength = 0.1
# custom_regularizer = partial(my_regularizer, strength=binary_reg_strength)
#
# def create_autoencoder_and_extract_models(sequence_length, learning_rate=0.001, binary_reg_strength=0.1):
#     input_layer = Input(shape=(sequence_length,))
#     encoder_hidden_1 = Dense(sequence_length // 2, activation='tanh')(input_layer)
#     encoder_hidden_2 = Dense(sequence_length // 4, activation='tanh')(encoder_hidden_1)
#     # Apply binary regularization to the latent layer
#     latent = Dense(1, activation='sigmoid', activity_regularizer=my_regularizer)(encoder_hidden_2)
#     encoder = Model(input_layer, latent)
#
#     encoded_input = Input(shape=(1,))
#     decoder_hidden_1 = Dense(sequence_length // 4, activation='tanh')(encoded_input)
#     decoder_hidden_2 = Dense(sequence_length // 2, activation='tanh')(decoder_hidden_1)
#     output = Dense(sequence_length, activation='tanh')(decoder_hidden_2)
#     decoder = Model(encoded_input, output)
#
#     autoencoder_model = Model(input_layer, decoder(encoder(input_layer)))
#     adam_optimizer = Adam(learning_rate=learning_rate)
#     autoencoder_model.compile(optimizer=adam_optimizer, loss='mean_squared_error')
#
#     return autoencoder_model, encoder, decoder

def train_autoencoder(train_data, validation_data, learning_rate=0.001):
    autoencoder, encoder, decoder = create_autoencoder_and_extract_models(sequence_length, learning_rate)
    autoencoder.summary()
    history = autoencoder.fit(train_data[:, :], train_data[:, :], epochs=1, batch_size=1, shuffle=False,
                              validation_data=(validation_data, validation_data))
    return autoencoder, encoder, decoder, history

def train_autoencoder_multimodal(in_data, lab_data, in_val_data, lab_val_data, learning_rate=0.001):
    autoencoder, encoder, decoder = create_autoencoder_and_extract_models(sequence_length, learning_rate)
    history = autoencoder.fit(in_data, lab_data,
                            epochs=1,
                            batch_size=1,
                            shuffle=False,
                            validation_data=(in_val_data, lab_val_data))
    return autoencoder, encoder, decoder, history

def create_dynamic_model(window_size=8, learning_rate=0.001):
    # Define a simple neural network model for binary classification
    input_layer = Input(shape=(window_size, ))
    # standardization_layer = Lambda(standardize_input)(input_layer)
    dense_1 = Dense(window_size // 2, activation='gelu')(input_layer)
    dense_2 = Dense(window_size // 4, activation='gelu')(dense_1)
    # dense_3 = Dense(window_size // 2, activation='relu')(dense_2)
    # dense_4 = Dense(window_size // 2, activation='relu')(dense_1)
    output = Dense(1, activation='gelu')(dense_1)
    model = Model(input_layer, output)
    adam_optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam_optimizer, loss='mean_squared_error', metrics=['accuracy'])  # binary_crossentropy
    return model

def train_dynamic_model(train_data, train_label, valid_data, valid_label, epochs=10, batch_size=32, lr=0.001):
    # Convert labels to categorical (one-hot encoding)
    train_label = np.mean(train_label, axis=1) # >= 0.5
    valid_label = np.mean(valid_label, axis=1) # >= 0.5
    # train_label_cat = to_categorical(train_label, num_classes=2)
    # valid_label_cat = to_categorical(valid_label, num_classes=2)
    train_label_cat = np.copy(train_label)
    valid_label_cat = np.copy(valid_label)
    model = create_dynamic_model(train_data.shape[1], learning_rate=lr)
    model.summary()

    # Initialize ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
    # Configure the ModelCheckpoint callback
    model_checkpoint_callback = ModelCheckpoint(
        filepath='best_model.h5',  # Specify the path to save the best model
        monitor='val_loss',  # Monitor the validation loss
        verbose=1,  # Verbosity mode
        save_best_only=True,  # Save only the best model
        mode='min'  # Minimize the validation loss
    )
    history = model.fit(train_data, train_label,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=False,
                        validation_data=(valid_data, valid_label),
                        callbacks=[reduce_lr, model_checkpoint_callback])

    return model, history

def load_data(directory_path):
    all_data_snc1 = []
    all_data_snc2 = []
    all_data_snc3 = []
    all_data_acc1 = []
    all_data_acc2 = []
    all_data_acc3 = []
    label_data = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):  # Ensure to load only CSV files
            file_path = os.path.join(directory_path, file_name)
            df = pd.read_csv(file_path)
            snc_1 = df['Snc1'].dropna().to_numpy()
            snc_2 = df['Snc2'].dropna().to_numpy()
            snc_3 = df['Snc3'].dropna().to_numpy()
            acc_1 = df['Acc1'].dropna().to_numpy()
            acc_2 = df['Acc2'].dropna().to_numpy()
            acc_3 = df['Acc3'].dropna().to_numpy()
            label = df['SncButton'].dropna().to_numpy()
            snc_1 = signal.resample(snc_1, len(acc_1), window='hamming')
            snc_2 = signal.resample(snc_2, len(acc_1), window='hamming')
            snc_3 = signal.resample(snc_3, len(acc_1), window='hamming')
            label = signal.resample(label, len(acc_1), window='hamming')
            all_data_snc1.append(snc_1 / np.max(snc_1))  # Keep as list of arrays
            all_data_snc2.append(snc_2 / np.max(snc_2))
            all_data_snc3.append(snc_3 / np.max(snc_3))
            all_data_acc1.append(acc_1 / np.max(acc_1))
            all_data_acc2.append(acc_2 / np.max(acc_2))
            all_data_acc3.append(acc_3 / np.max(acc_3))
            label_data.append(label)

    return all_data_snc1, all_data_snc2, all_data_snc3, all_data_acc1, all_data_acc2, all_data_acc3, label_data

def extract_sequences(data, label_data, sequence_length=sequence_length):
    sequences = []
    label_sequences = []
    series_num = 0
    for series in data:
        for i in range(0, len(series), sequence_length):
            # Extract a sequence of 'sequence_length' from the time series
            sequence = series[i:i + sequence_length]
            label_sequence = label_data[series_num][i:i + sequence_length]
            # Only add sequences that are exactly 'sequence_length' long
            if len(sequence) == sequence_length:
                sequences.append(sequence)
                label_sequences.append(label_sequence)
        series_num += 1
    return np.array(sequences), np.array(label_sequences)

def running_average_with_memory(data, window_size=200):
    averages = [0] * len(data)  # Initialize the averages list with zeros
    for i in range(len(data)):
        start_index = max(0, i - window_size - 1)  # Ensure memory of 200 samples
        current_window = data[start_index:i + 1]
        window_average = np.mean(current_window)
        averages[i] = window_average
    return averages

def split_dataset(data, label_data, train_size=0.6, test_size=0.2):
    # Split data into training and temp (validation + test)
    train_data, temp_data = train_test_split(data, train_size=train_size, shuffle=False)
    train_label, temp_label = train_test_split(label_data, train_size=train_size, shuffle=False)
    # Split temp into validation and test
    validation_data, test_data = train_test_split(temp_data, test_size=test_size / (1 - train_size), shuffle=False)
    validation_label, test_label = train_test_split(temp_label, test_size=test_size / (1 - train_size), shuffle=False)
    return train_data, validation_data, test_data, train_label, validation_label, test_label

def jacobian_tensorflow(x, model, verbose=False):
    # Ensure x is a TensorFlow tensor
    x = tf.convert_to_tensor(x.reshape((1, -1)), dtype=tf.float32)

    # Get the output dimension
    O = model.output_shape[-1]

    # Initialize jacobian matrix
    jacobian_matrix = []

    # Iteration range with tqdm for progress bar if verbose
    it = tqdm(range(O)) if verbose else range(O)

    for o in it:
        with tf.GradientTape() as tape:
            tape.watch(x)
            output = model(x)
            loss = output[0, o]

        gradients = tape.gradient(loss, x)
        jacobian_matrix.append(gradients.numpy().flatten())

    return np.array(jacobian_matrix)

def forward_propagation(x_k, dt=int(1/1060)):
    f = np.array([[1, dt], [0, 1]]).reshape((2, 2))
    x_kp1 = np.dot(f, x_k)
    return x_kp1

def observation_calculation(observation_model, x_k):
    return observation_model(x_k)

def calculate_velocity(x_kp1, x_k):
    if x_kp1 > x_k:
        return 1.0
    elif x_kp1 == x_k:
        return 0.0
    else:
        return -1.0

def saturate_state(x_k):
    if x_k >= 1.0:
        return 1.0
    else:
        return 0.0


# Predict on new data
def simulate_additive_noise(signal, num_windows, min_noise_duration=10, max_noise_duration=20):
    """
    Simulate additive noise with saturation between -1 and 1.

    Parameters:
    - signal: The original signal array.
    - num_windows: Total number of data windows in the signal.
    - min_noise_duration: Minimum duration of noise in data windows.
    - max_noise_duration: Maximum duration of noise in data windows.

    Returns:
    - The signal with additive noise.
    """
    # Initialize the noisy signal with the original signal
    noisy_signal = np.copy(signal)

    current_window = 0
    while current_window < num_windows:
        # Randomly choose the duration for the current noise
        noise_duration = np.random.randint(min_noise_duration, max_noise_duration + 1)

        # Ensure the noise duration does not exceed the signal length
        if current_window + noise_duration > num_windows:
            noise_duration = num_windows - current_window

        # Draw a random saturation value for each window within the noise duration
        for i in range(noise_duration):
            saturation_value = np.random.uniform(-1, 1)
            if saturation_value >= 0:
                saturation_value = 10
            else:
                saturation_value = -10
            # Apply the noise and clip the signal to maintain saturation limits
            noisy_signal[current_window + i] = np.clip(noisy_signal[current_window + i] + saturation_value, -1, 1)

        # Move to the next window after the noise duration
        current_window += noise_duration

    return noisy_signal
