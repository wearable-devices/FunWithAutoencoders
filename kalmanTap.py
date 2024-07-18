import numpy as np
import matplotlib.pyplot as plt
import autoencoder_auxiliary as aux
from sklearn.cluster import KMeans
import scipy.signal as signal
import visualizations as viz
import time as time
from autoencoder_auxiliary import sequence_length

input_directory_path = '/Users/anton.k/Documents/Data/Train_edited/AsherDahan/tap' # DragNDropSwipeFling/Feature_Extraction_playground
# input_directory_path = '/Users/anton.k/Documents/Data/Train_edited/AntonKuper/Tap' # DragNDropSwipeFling/Feature_Extraction_playground
# input_directory_path = '/Users/anton.k/Documents/Data/Train_edited/LeeorLanger/Tap' # DragNDropSwipeFling/Feature_Extraction_playground
# input_directory_path = '/Users/anton.k/Documents/Data/Train_edited/LeeorLanger/Tap' # DragNDropSwipeFling/Feature_Extraction_playground
# input_directory_path = '/Users/anton.k/Documents/Data/Train_edited/ItaiAronsberg/Tap' # DragNDropSwipeFling/Feature_Extraction_playground
segment_length = sequence_length
# Load and preprocess the dataset
snc_1, snc_2, snc_3, acc_1, acc_2, acc_3, label_data = aux.load_data(input_directory_path)
sos = signal.butter(4, 2, 'highpass', output='sos', fs=1125)
acc_1_f = []
acc_2_f = []
acc_3_f = []
for ser1, ser2, ser3 in zip(acc_1, acc_2, acc_3):
    acc_1_f.append(signal.sosfiltfilt(sos, np.array(ser1)).tolist())
    acc_2_f.append(signal.sosfiltfilt(sos, np.array(ser2)).tolist())
    acc_3_f.append(signal.sosfiltfilt(sos, np.array(ser3)).tolist())
sequenced_data, sequenced_label = aux.extract_sequences(snc_1, label_data)
sequenced_data_2, _ = aux.extract_sequences(snc_2, label_data)
sequenced_data_3, _ = aux.extract_sequences(snc_3, label_data)
sequenced_data_acc1, _ = aux.extract_sequences(acc_1_f, label_data)
sequenced_data_acc2, _ = aux.extract_sequences(acc_2_f, label_data)
sequenced_data_acc3, _ = aux.extract_sequences(acc_3_f, label_data)
# Split the dataset
train_data, validation_data, test_data, train_label, validation_label, test_label = aux.split_dataset(sequenced_data, sequenced_label)
train_data, validation_data, test_data, _, _, _ = aux.split_dataset(sequenced_data_2, sequenced_label)
train_data_3, validation_data_3, test_data_3, _, _, _ = aux.split_dataset(sequenced_data_3, sequenced_label)
train_data_acc, validation_data_acc, test_data_acc, _, _, _ = aux.split_dataset(sequenced_data_acc1, sequenced_label)
train_data_acc2, validation_data_acc2, test_data_acc2, _, _, _ = aux.split_dataset(sequenced_data_acc2, sequenced_label)
train_data_acc3, validation_data_acc3, test_data_acc3, _, _, _ = aux.split_dataset(sequenced_data_acc3, sequenced_label)
# Train the autoencoder on clean data
print("Training Clean Model")
autoencoder, encoder, decoder, history = aux.train_autoencoder(train_data, validation_data, learning_rate=0.001)
# J_c = aux.jacobian_tensorflow(encoder.predict(test_data[1, :].reshape(1, -1)), decoder, verbose=False)
# print(J_c.shape)
# autoencoder_2, encoder_2, decoder_2, history_2 = aux.train_autoencoder(train_data_2, validation_data_2, learning_rate=0.001)
# autoencoder_3, encoder_3, decoder_3, history_3 = aux.train_autoencoder(train_data_3, validation_data_3, learning_rate=0.001)
# autoencoder, encoder, decoder, history = aux.train_autoencoder_multimodal(train_data_acc, train_data,
#                                                                           validation_data_acc, validation_data, learning_rate=0.0001)
# f_model, f_history = aux.train_dynamic_model(train_data_acc, train_label, validation_data_acc, validation_label,
#                                              epochs=100, batch_size=128, lr=0.001)
# J_f = aux.jacobian_tensorflow(f_model.predict(test_data_acc2[1, :].reshape(1, -1)), f_model, verbose=False)
# print(J_f.shape)

# # Evaluate the model on the test set
# print("Evaluating Clean Model")
# test_loss = autoencoder.evaluate(test_data, test_data, batch_size=1)
# print(f'Test Loss: {test_loss}')

reconstructed_data = autoencoder.predict(test_data, verbose=0)
encoded_test_data = encoder.predict(test_data, verbose=0)
encoded_train_data = encoder.predict(train_data, verbose=0)
# Concatenate all samples to form continuous series
test_data_series = np.concatenate(test_data)
reconstructed_data_series = np.concatenate(reconstructed_data)
encoded_data_series_1 = np.concatenate(encoded_test_data[:, 0].reshape(-1, 1))
encoded_data_series_2 = np.concatenate(encoded_test_data[:, 1].reshape(-1, 1))
test_label_series = np.concatenate(test_label)
# Clustering in the latent space
kmeans = KMeans(n_clusters=1, random_state=0)
kmeans.fit(encoded_train_data)
# Add noise to test data set
test_data_noisy = np.copy(test_data)  # test_label
num_segments = 50  # Number of segments to inject noise into
outlier_indices = np.random.choice(test_data.shape[0], num_segments, replace=False)
for start_index in outlier_indices:
    test_data_noisy[start_index, :int(segment_length // 2)] += 1
    test_data_noisy[start_index, int(segment_length // 2):] -= 1
    # test_data_noisy[start_index, :] -= np.random.random(segment_length)
test_data_noisy_series = np.concatenate(test_data_noisy)
encoded_test_data = encoder.predict(test_data_noisy, verbose=0)
encoded_train_data = encoder.predict(train_data, verbose=0)
kmeans_test_labels = kmeans.predict(encoded_test_data)
kmeans_train_labels = kmeans.predict(encoded_train_data)
# f_predictions = f_model.predict(test_data_acc)
# decoder_output = decoder.predict(f_predictions, verbose=0)
# decoder_output_series = np.concatenate(decoder_output)
# f_predictions_series = np.concatenate(f_predictions)
label_series = np.concatenate(test_label)
# f_predictions_series = signal.resample(f_predictions_series, len(label_series), axis=0, window='hamming')

# Visualizations
# viz.plot_data_comparison(test_label_series, test_data_series, reconstructed_data_series,
#                          signal.resample(f_predictions_series, len(encoded_data_series_1)),
#                          encoded_data_series_2)
# f_predictions_series = signal.sosfiltfilt(sos, f_predictions_series)
# viz.plot_observation_model(test_data_series, reconstructed_data_series, decoder_output_series,
#                            signal.resample(f_predictions_series, len(encoded_data_series_1)), encoded_data_series_1)
# viz.plot_f_label_comparison(label_series, f_predictions_series, test_data_series - decoder_output_series)
viz.plot_autoencoder_results(autoencoder, test_data_noisy, test_data_noisy_series)
# viz.plot_clustering_with_noise(encoded_train_data, encoded_test_data, kmeans_train_labels, kmeans_test_labels, segment_starts)
viz.plot_clustering_with_outliers(encoded_test_data, kmeans_test_labels, outlier_indices)
plt.show()

#################   Future Work ##################
# Train the autoencoder on noisy data
train_data_noisy = np.copy(train_data)
validation_data_noisy = np.copy(validation_data)
num_segments = 150  # Number of segments to inject noise into
segment_length = train_data.shape[1]  # Define the length of each segment
# Randomly select num_segments unique starting indices for the segments
segment_starts_tr = np.random.choice(train_data.shape[0], num_segments, replace=False)
i = 0
for start_index in segment_starts_tr:
    train_data_noisy[start_index, :int(segment_length // 2)] += 1 # np.random.random(segment_length)
    train_data_noisy[start_index, int(segment_length // 2):] -= 1
    # train_data_noisy[start_index, :] -= np.random.random(segment_length)
num_segments = 50  # Number of segments to inject noise into
segment_starts_val = np.random.choice(validation_data.shape[0], num_segments, replace=False)
for start_index in segment_starts_val:
    validation_data_noisy[start_index, :int(segment_length // 2)] += 1 #np.random.random(segment_length)
    validation_data_noisy[start_index, int(segment_length // 2):] -= 1 #np.random.random(segment_length)
    # validation_data_noisy[start_index, :] -= np.random.random(segment_length)

print("Training Noisy Model")
autoencoder_noisy, encoder_noisy, decoder_noisy, history = aux.train_autoencoder(train_data_noisy, validation_data_noisy, learning_rate=0.001)
reconstructed_data_noisy_ae = autoencoder_noisy.predict(test_data_noisy, verbose=0)
encoded_test_data_noisy_ae = encoder_noisy.predict(test_data_noisy, verbose=0)
encoded_train_data_noisy_ae = encoder_noisy.predict(train_data_noisy, verbose=0)
# Concatenate all samples to form continuous series
reconstructed_data_noisy_ae_series = np.concatenate(reconstructed_data_noisy_ae)
encoded_test_data_series_1_noisy_ae = np.concatenate(encoded_test_data_noisy_ae[:, 0].reshape(-1, 1))
encoded_test_data_series_2_noisy_ae = np.concatenate(encoded_test_data_noisy_ae[:, 1].reshape(-1, 1))
encoded_train_data_series_1_noisy_ae = np.concatenate(encoded_train_data_noisy_ae[:, 0].reshape(-1, 1))
encoded_train_data_series_2_noisy_ae = np.concatenate(encoded_train_data_noisy_ae[:, 1].reshape(-1, 1))
# Clustering in the latent space
kmeans_noisy_ae = KMeans(n_clusters=1, random_state=0)
kmeans_noisy_ae.fit(encoded_train_data_noisy_ae)
kmeans_test_labels_noisy_ae = kmeans.predict(encoded_test_data_noisy_ae)
kmeans_train_labels_noisy_ae = kmeans.predict(encoded_train_data_noisy_ae)
viz.plot_autoencoder_results(autoencoder_noisy, test_data_noisy, test_data_noisy_series)
# viz.plot_clustering_with_noise(encoded_train_data_noisy_ae, encoded_test_data_noisy_ae, kmeans_train_labels_noisy_ae, kmeans_test_labels_noisy_ae, segment_starts)
viz.plot_clustering_with_outliers(encoded_test_data_noisy_ae, kmeans_test_labels_noisy_ae, outlier_indices)
plt.show()