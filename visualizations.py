import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve
def plot_sensor_data(sensor_data, button_data, plot_acc=False, plot_gyr=False, plot_snc=False, title_str=''):
    # data is a numpy array of shape (num_samples, num_sensors*3)
    # it should contain sensor data from only the sensors which you want to plot
    sensors_to_plot = int(plot_acc) + int(plot_gyr) + int(plot_snc)
    fig, ax = plt.subplots(sensors_to_plot, 3)
    for i in range(sensors_to_plot):
        for j in range(3):
            ax[i, j].plot(sensor_data[:, i*3+j])
            ax[i, j].plot(button_data, '--')
            ax[i, j].set_title(title_str)


def plot_correlations(data, labels):
    # Visualize correlation matrix
    means, stds, cor_mat = fe.statistical_features(data)
    for i in range(len(cor_mat)):
        cor_mat[i, i] = 0
    plt.matshow(cor_mat)
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    plt.colorbar()


def plot_processed_snc_signals(snc_sig_raw, label_hist, filtered_signal_w, scaled_hist_pn, filtered_signal_pnw):
    gaussian_kernel = fe.calculate_gaussian_kernel(sigma=100, window_size=128)
    fac = 2
    fac_weiner = fac**8
    fac_pink = fac
    fac_pink_weiner = fac**10
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 8), tight_layout=True)

    plt.subplot(3, 1, 1)
    plt.plot(snc_sig_raw, 'b', label='Raw')
    plt.plot(np.linspace(0, len(snc_sig_raw), len(filtered_signal_w)), filtered_signal_w, 'g', linewidth=2,
             label='Weiner Filtered')
    filtered_signal_w = fac_weiner * convolve(filtered_signal_w**2, gaussian_kernel, mode='same')
    import scipy.signal as signal
    filtered_signal_w = signal.resample(filtered_signal_w, len(label_hist))
    # filtered_signal_w = filtered_signal_w > 0.025
    plt.plot(np.linspace(0, len(snc_sig_raw), len(filtered_signal_w)), filtered_signal_w, 'y', linewidth=2,
             label='Weiner Filtered')
    from sklearn.metrics import confusion_matrix, accuracy_score
    filtered_signal_w = filtered_signal_w > 0.02
    # print("Accuracy: ", accuracy_score(label_hist, filtered_signal_w))
    # print('Confusion Matrix: \n', confusion_matrix(label_hist, filtered_signal_w, normalize='all'))
    plt.plot(label_hist, 'w', linewidth=2)
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.grid()
    plt.title('Weiner Filtered (Swipe Left/Right, Bend up/Down)')

    plt.subplot(3, 1, 2)
    plt.plot(snc_sig_raw, 'b', label='Raw')
    plt.plot(scaled_hist_pn, 'g', linewidth=2, label='Raw Scaled by Noise')
    scaled_hist_pn = fac_pink * convolve(scaled_hist_pn**2, gaussian_kernel, mode='same')
    plt.plot(scaled_hist_pn, 'y', linewidth=2, label='Raw Scaled by Noise')
    plt.plot(label_hist, 'w', linewidth=2)
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.grid()
    plt.title('Pink Noise Scaled (Swipe Left/Right, Bend up/Down)')

    plt.subplot(3, 1, 3)
    plt.plot(snc_sig_raw, 'b', label='Raw')
    plt.plot(np.linspace(0, len(snc_sig_raw), len(filtered_signal_pnw)), filtered_signal_pnw, 'g', linewidth=2,
             label='Pinky Scaled --> Weiner Filtered')
    filtered_signal_pnw = fac_pink_weiner * convolve(filtered_signal_pnw**2, gaussian_kernel, mode='same')
    plt.plot(np.linspace(0, len(snc_sig_raw), len(filtered_signal_pnw)), filtered_signal_pnw, 'y', linewidth=2,
             label='Pinky Scaled --> Weiner Filtered')
    plt.plot(label_hist, 'w', linewidth=2)
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.grid()
    plt.title('Pink Noise Scaled Then Weiner Filtered (Swipe Left/Right, Bend up/Down)')
    plt.subplots_adjust(hspace=0.5, wspace=0.3)


def plot_snc_decomposition(sig_raw, label, comp1, comp2, comp3, name):
    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(sig_raw, 'b', label='Raw')
    plt.plot(comp1, 'g', label=name + ' 1')
    plt.plot(label, 'w', linewidth=2)
    plt.title(name)
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(sig_raw, 'b', label='Raw')
    plt.plot(comp2, 'g', label=name + ' 2')
    plt.plot(label, 'w', linewidth=2)
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(sig_raw, 'b', label='Raw')
    plt.plot(comp3, 'g', label=name + ' 3')
    plt.plot(label, 'w', linewidth=2)
    plt.legend()
    plt.grid()

def plot_clustering(encoded_data, cluster_labels):
    plt.figure(figsize=(8, 6))
    plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
    plt.title('Clustering of Latent Vectors')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.colorbar(label='Cluster')
    plt.show()

def plot_data_comparison(test_label_series, test_data_series, reconstructed_data_series, encoded_data_series_1, encoded_data_series_2):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot((test_label_series > 0.5) * 0.5, label='Label', alpha=0.7, color='black', linewidth=1)
    plt.plot(test_data_series, label='Original Data', alpha=0.7)
    plt.plot(reconstructed_data_series, label='Reconstructed Data', alpha=0.7)
    # Uncomment the following lines if needed
    # plt.plot(test_data_series - reconstructed_data_series, label='Error', alpha=0.7, color='green', linewidth=1)
    # plt.plot(test_data_series_noisy, label='Noisy', alpha=0.7, color='red', linewidth=1)
    plt.title('Comparison of Original and Reconstructed Data')
    plt.legend()

    plt.subplot(2, 1, 2)
    # Uncomment and replace `history.history['loss']` and `history.history['val_loss']` with appropriate variables if needed
    # plt.plot(history.history['loss'], label='Loss', alpha=0.7)
    # plt.plot(history.history['val_loss'], label='Validation Loss', alpha=0.7)
    # plt.plot((test_label > 0.5) * 0.5, label='Label', alpha=0.7, color='black', linewidth=1)
    # plt.plot(signal.resample(encoded_data_series_1, len(test_data_series), window='hamming'), label=r'Latent z1', alpha=0.5)
    # plt.plot(signal.resample(encoded_data_series_2, len(test_data_series), window='hamming'), label=r'Latent z2', alpha=0.5)
    # plt.plot((test_label_series > 0.5) * 0.5, label='Label', alpha=0.7, color='black', linewidth=1)
    import scipy.signal as signal
    # plt.plot(signal.resample(abs(encoded_data_series_1) > 0.1, len(test_label_series), window='hamming'), label='Encoded Data 1', alpha=0.7)
    # plt.plot(signal.resample(abs(encoded_data_series_2) > 0.1, len(test_label_series), window='hamming'), label='Encoded Data 2', alpha=0.7)
    plt.plot((signal.resample(test_label_series, len(encoded_data_series_1)) > 0.5) * 0.5, label='Label', alpha=0.7, color='black', linewidth=1)
    plt.plot(encoded_data_series_1, label='Encoded Data 1', alpha=0.7)
    plt.plot(encoded_data_series_2, label='Encoded Data 2', alpha=0.7)

    plt.title('Evolution of Loss and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()

def plot_clustering_with_noise(encoded_train_data, encoded_test_data, kmeans_train_labels, kmeans_test_labels, segment_starts):
    plt.figure(figsize=(8, 6))
    plt.scatter(encoded_test_data[:, 0], encoded_test_data[:, 1], c=kmeans_test_labels, cmap='viridis', alpha=0.5)
    plt.title('Clustering of Test Data Latent Vectors')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.colorbar(label='Cluster')

    plt.figure(figsize=(10, 8))
    print(encoded_train_data.shape, kmeans_train_labels.shape)
    plt.scatter(encoded_train_data[:, 0], encoded_train_data[:, 1], c=kmeans_train_labels, cmap='viridis', alpha=0.5, marker='o',
                label='Train Data')
    plt.scatter(encoded_test_data[:, 0], encoded_test_data[:, 1], c=kmeans_test_labels, cmap='viridis', alpha=0.5,
                marker='x', label='Test Data')
    plt.scatter(encoded_test_data[segment_starts, 0], encoded_test_data[segment_starts, 1], c='red', marker='D',
                label='Noisy Test Data')

    plt.title('Clustering of Train and Test Data Latent Vectors')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.legend()
    plt.colorbar(label='Cluster')

def plot_autoencoder_results(autoencoder, test_data_noisy, test_data_series_noisy):
    plt.figure(figsize=(12, 6))
    # plt.subplot(2, 1, 1)
    # test_loss = autoencoder.evaluate(test_data_noisy, test_data_noisy, batch_size=1)
    # print(f'"Clean" Model, Noisy Test Set. Test Loss: {test_loss}')
    reconstructed_data = autoencoder.predict(test_data_noisy, verbose=0)
    reconstructed_data_series = np.concatenate(reconstructed_data)
    plt.plot(test_data_series_noisy, label='Noisy Data Ground Truth', alpha=0.5, color='blue', linewidth=1)
    plt.plot(reconstructed_data_series, label='Noisy Data Reconstructed.', alpha=0.5, color='green', linewidth=1)
    # plt.plot(test_data_series_noisy - reconstructed_data_series, label='Clean Model Prediction Error', alpha=0.7,
    #          color='black', linewidth=1, linestyle='--')
    # plt.plot(running_average_with_memory(test_data_series_noisy - reconstructed_data_series, window_size=50),
    #          label='Clean Model Prediction Error', alpha=0.7, color='red', linewidth=1, linestyle='--')
    plt.legend()
    # plt.subplot(2, 1, 2)
    # test_loss = autoencoder_noisy.evaluate(test_data_noisy, test_data_noisy, batch_size=1)
    # print(f'"Noisy" Model, Noisy Test Set. Test Loss: {test_loss}')
    # plt.plot(test_data_series_noisy, label='Noisy Data', alpha=0.5, color='blue', linewidth=1)
    # plt.plot(reconstructed_data_series_noisy, label='Noisy Data Pred.', alpha=0.5, color='green', linewidth=1)
    # plt.plot(test_data_series_noisy - reconstructed_data_series_noisy, label='Noisy Model Prediction Error', alpha=0.7,
    #          color='black', linewidth=1, linestyle='--')
    # plt.plot(running_average_with_memory(test_data_series_noisy - reconstructed_data_series_noisy, window_size=50),
    #          label='Noisy Model Prediction Error', alpha=0.7, color='red', linewidth=1, linestyle='--')
    # plt.legend()
    plt.tight_layout()

def plot_f_label_comparison(label_series, predictions_series_acc, predictions_series_tot):
    plt.figure(figsize=(12, 6))
    plt.plot(label_series >= 0.5, label='Label', alpha=0.7, color='black', linewidth=1)
    plt.plot(predictions_series_acc >= 0.5, label='Prediction Acc', alpha=0.8)
    gain = 0.5
    plt.plot((predictions_series_acc + gain*predictions_series_tot) >= 0.5, linestyle ='--',
             label='Prediction Kalman', alpha=0.4)
    plt.title('Comparison of Original and Predicted Labels')
    plt.legend()
    plt.grid()
    plt.xlabel('Sample')
    plt.show()


def plot_observation_model(original_data, reconstructed_data, reconstructed_from_acc_data, acc_pred, latent_snc):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(original_data, label='original', alpha=0.8, color='black', linewidth=1)
    plt.plot(reconstructed_data, label='reconstructed snc', alpha=0.7)
    plt.plot(reconstructed_from_acc_data, label='reconstructed acc', alpha=0.3)
    plt.title('Comparison of Original, Reconstructed Data and decoded from classifier')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(acc_pred, label='prediction acc classifier', alpha=0.7)
    plt.plot(latent_snc, label='latent snc autoencoder', alpha=0.7)
    plt.title('Classifier prediction vs Autoencoder latent space')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()


def plot_clustering_with_outliers(encoded_test_data, cluster_labels, outliers_indices):
    """
    Plot clustering results on the latent space of the test set and mark outliers with red diamonds.
    Parameters:
    - encoded_test_data: 2D numpy array of the encoded test data.
    - cluster_labels: Array of cluster labels corresponding to each point in encoded_test_data.
    - outliers_indices: List or array of indices of outliers in encoded_test_data.
    This function assumes that the latent space is 2-dimensional.
    """
    plt.figure(figsize=(8, 6))
    # Plot all points in the test set
    plt.scatter(encoded_test_data[:, 0], encoded_test_data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
    # Highlight outliers with red diamonds
    plt.scatter(encoded_test_data[outliers_indices, 0], encoded_test_data[outliers_indices, 1], c='red', marker='D', s=50)
    plt.title('Clustering of Test Data Latent Vectors with Outliers')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.colorbar(label='Cluster')
    plt.show()