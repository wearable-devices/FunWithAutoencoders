from kalmanTap import aux, viz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
def main():
    # Load or generate the necessary data
    train_data, validation_data, test_data_noisy, test_data, test_label = aux.load_data()

    # Inject noise into training and validation data
    train_data_noisy, validation_data_noisy = aux.inject_noise(train_data, validation_data)

    # Train the autoencoder on noisy data
    autoencoder_noisy, encoder_noisy, decoder_noisy, history = aux.train_autoencoder(train_data_noisy, validation_data_noisy, learning_rate=0.001)

    # Predict using the noisy autoencoder
    test_data_noisy_series = np.concatenate(test_data_noisy)
    viz.plot_autoencoder_results(autoencoder_noisy, test_data_noisy, test_data_noisy_series)

    # Perform clustering and plot with outliers
    encoded_test_data_noisy_ae = encoder_noisy.predict(test_data_noisy)
    kmeans = KMeans(n_clusters=1, random_state=0).fit(encoded_test_data_noisy_ae)
    kmeans_test_labels_noisy_ae = kmeans.predict(encoded_test_data_noisy_ae)
    outlier_indices = aux.detect_outliers(encoded_test_data_noisy_ae, kmeans_test_labels_noisy_ae)
    viz.plot_clustering_with_outliers(encoded_test_data_noisy_ae, kmeans_test_labels_noisy_ae, outlier_indices)

    plt.show()

if __name__ == '__main__':
    main()


