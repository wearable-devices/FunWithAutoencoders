import numpy as np
import scipy.signal as signal

def statistical_features(data):
    # data contains sensor readings, its dimas are (n_samples x n_sensors)
    means = data.mean(axis=0)
    stds = data.std(axis=0, ddof=1)
    corr_mat = np.corrcoef(data, rowvar=False)
    return means, stds, corr_mat

def pink(f):
    return 1/np.where(f == 0, float('inf'), f)


def pink_noise_scaling(snc_signal, buffer_size):
    X_freq_domain = np.fft.rfft(snc_signal)
    scaling = pink(np.fft.rfftfreq(buffer_size))
    # # Normalize S (optional)
    scaling = scaling / np.sqrt(np.mean(scaling**2))
    X_freq_domain = X_freq_domain * scaling
    X_time_domain = np.fft.irfft(X_freq_domain)
    return X_time_domain


def calc_pink_noise_scaling(data_loader, buffer_size, to_scale=False):
    eof = False
    snc_sig_raw = np.array([])
    scaled_hist = np.array([])
    data_loader.buffer_count = 0
    data_loader.end_of_file = False
    while not eof:
        data_loader.next_buffer(buffer_size=buffer_size)
        snc_signal, label, eof = data_loader.get_snc_from_buffer(to_scale=to_scale)
        snc_sig_raw = np.append(snc_sig_raw, snc_signal[:, 1])
        snc_scaled = pink_noise_scaling(snc_signal[:, 1], buffer_size)
        scaled_hist = np.append(scaled_hist, snc_scaled)

    return scaled_hist


def calculate_gaussian_kernel(sigma=50, window_size=64):
    # Assuming sigma (standard deviation) for the Gaussian filter
    # Generate a Gaussian kernel
    gaussian_kernel = np.exp(-0.5 * (np.arange(-(window_size - 1) / 2, (window_size - 1) / 2 + 1) / sigma) ** 2)
    gaussian_kernel /= np.sum(gaussian_kernel)

    return gaussian_kernel


def calc_wavefront_length(data_loader, buffer_size, to_scale=False):
    eof = False
    snc_sig_raw = np.array([])
    snc_wl_hist = np.array([])
    data_loader.buffer_count = 0
    data_loader.end_of_file = False
    while not eof:
        data_loader.next_buffer(buffer_size=buffer_size)
        # data_loader.next_buffer_continuous(buffer_size=128, stride_size=10)
        snc_signal, label, eof = data_loader.get_snc_from_buffer(to_scale=to_scale)
        snc_sig_raw = np.append(snc_sig_raw, snc_signal[:, 1])
        snc_wl = np.sum(abs(snc_signal[1:, 1] - snc_signal[:-1, 1]))
        snc_wl_hist = np.append(snc_wl_hist, snc_wl)

    return snc_wl_hist


def sampen(data, m, r):
    """
    Computes Sample Entropy (SampEn) of a time series data.
    m: Length of compared run of data
    r: Tolerance (typically 0.2 * std(data))
    """
    N = len(data)

    # Split time series and save all templates of length m and m+1
    xmi = np.array([data[i:i + m] for i in range(N - m)])
    xmj = np.array([data[i:i + m + 1] for i in range(N - m - 1)])
    print(N, data.shape, xmj.shape, xmi.shape)

    B = 0.0
    A = 0.0

    # Calculate the distance between each template of length m and all other templates of length m
    for i in range(len(xmi)):
        for j in range(i + 1, len(xmi)):
            if np.linalg.norm(xmi[i] - xmi[j], ord=np.inf) <= r:
                B += 1
    # Calculate the distance between each template of length m+1 and all other templates of length m+1
    for i in range(len(xmj)):
        for j in range(i + 1, len(xmj)):
            if np.linalg.norm(xmj[i] - xmj[j], ord=np.inf) <= r:
                A += 1
    # Normalizing the counts
    B /= (N - m) * (N - m - 1) / 2
    A /= (N - m - 1) * (N - m - 2) / 2

    if A == 0:
        return np.inf

    return -np.log(A / B)


def calc_sampl_enthropy(data_loader, m, r, buffer_size=128):
    eof = False
    snc_sig_raw = np.array([])
    snc_enthropy_hist = np.array([])
    data_loader.buffer_count = 0
    data_loader.end_of_file = False
    while not eof:
        data_loader.next_buffer(buffer_size=buffer_size)
        # data_loader.next_buffer_continuous(buffer_size=128, stride_size=10)
        snc_signal, label, eof = data_loader.get_snc_from_buffer(to_scale=False)
        snc_signal -= np.mean(snc_signal, axis=0)
        snc_sig_raw = np.append(snc_sig_raw, snc_signal[:, 1])
        snc_enthropy = 1 * sampen(snc_signal[:, 0], m, r) + 0 * sampen(snc_signal[:, 1], m, r) + 0 * sampen(snc_signal[:, 2], m, r)
        snc_enthropy_hist = np.append(snc_enthropy_hist, snc_enthropy)

    return snc_enthropy_hist


def higuchi_fd(x, kmax):
    """
    Computes Higuchi's Fractal Dimension (HFD) of a time series x.
    kmax: Maximum interval of k.
    """
    L = []
    x = np.array(x)
    N = len(x)

    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            Lmk = 0
            for i in range(1, int(np.floor((N - m) / k))):
                Lmk += abs(x[m + i * k] - x[m + (i - 1) * k])
            Lmk = (Lmk * (N - 1) / int(np.floor((N - m) / k)) / k)
            Lk.append(Lmk)
        L.append(np.log(np.mean(Lk)))

    lk = np.log(1.0 / np.arange(1, kmax + 1))
    HFD = np.polyfit(lk, L, 1)[0]
    return HFD

# def higuchi_fd(X, kmax):
#     L = []
#     x = X
#     N = len(x)
#     for k in range(1, kmax + 1):
#         Lk = np.zeros(k)
#         for m in range(k):
#             Lmk = 0
#             for i in range(1, int(np.floor((N - m) / k))):
#                 Lmk += np.abs(x[m + i * k] - x[m + i * k - k])
#             Lmk = Lmk * (N - 1) / np.floor((N - m) / k) / k
#             Lk[m] = Lmk
#         L.append(np.log(np.mean(Lk)))
#     return np.polyfit(np.log(np.arange(1, kmax + 1)), L, 1)[0]




def calculate_higuchi_fractal_dimension(data_loader, buffer_size):
    eof = False
    snc_sig_raw = np.array([])
    higuchi_dim_hist = np.array([])
    data_loader.buffer_count = 0
    data_loader.end_of_file = False
    while not eof:
        data_loader.next_buffer(buffer_size=buffer_size)
        # data_loader.next_buffer_continuous(buffer_size=128, stride_size=10)
        snc_signal, label, eof = data_loader.get_snc_from_buffer(to_scale=False)
        snc_signal -= np.mean(snc_signal, axis=0)
        snc_sig_raw = np.append(snc_sig_raw, snc_signal[:, 1])
        higuchi_dim = higuchi_fd(snc_signal[:, 1], kmax=5)
        higuchi_dim_hist = np.append(higuchi_dim_hist, higuchi_dim)

    return higuchi_dim_hist


def zero_order_hold_interpolation(Z, new_length):
    """
    Stretch the array Z to new_length using zero-order hold interpolation.
    """
    original_indices = np.arange(len(Z))
    stretched_indices = np.linspace(0, len(Z) - 1, new_length)
    stretched_Z = np.round(stretched_indices).astype(int)
    return Z[stretched_Z]




















