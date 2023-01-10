import os
import numpy as np
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor

from utils import get_max_shape, get_n_channels, get_dtype_range


def mean_absolute_error(img1, img2):
    # MAE = 1 / N * sum(|x - y|)
    abs_diff = np.abs(np.subtract(img1, img2, dtype=np.float64))
    mae = np.mean(abs_diff)

    return mae


def mean_squared_error(img1, img2):
    # MSE = 1 / N * sum((x - y) ** 2)
    diff = np.subtract(img1, img2, dtype=np.float64)
    mse = np.mean(np.square(diff))

    return mse


def peak_signal_noise_ratio(img1, img2):
    # PSNR = 10 * log10((max_i ** 2) / MSE)
    mse = mean_squared_error(img1, img2)

    if mse == 0:
        psnr = float('inf')
    else:
        max_i = get_dtype_range(img1.dtype.type)[1]
        psnr = 10 * np.log10(np.square(max_i) / mse)

    return psnr


# TODO: fix implementation, and implement Lewis numerator fft method
def normalized_cross_correlation(img1, img2):
    # Compute FFTs
    fft_x = np.fft.fft2(img1)
    fft_y = np.fft.fft2(img2)

    # Compute cross correlation
    cross_corr = np.multiply(fft_x, np.conj(fft_y))

    # Compute NCC
    ncc = np.real(np.fft.ifft2(cross_corr))

    # Normalize NCC
    ncc /= np.sqrt(np.sum(np.square(img1)) * np.sum(np.square(img2)))

    mean_ncc = 1 - np.mean(ncc)
    max_ncc = 1 - np.max(ncc)

    return (mean_ncc, max_ncc)


# TODO: fix implementation
def structural_similarity_index(img1, img2):
    # SSIM = l * c * s
    # where:
    # l = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    # c = (2 * sigma_x * sigma_y + C2) / (sigma_x ** 2 + sigma_y ** 2 + C2)
    # s = (sigma_xy + C3) / (sigma_x * sigma_y + C3)
    # as per Wang et. al. 2004.

    size = get_max_shape(img1, img2)
    n_channels = get_n_channels(img1)
    dmin, dmax = get_dtype_range(img1.dtype.type)
    dr = dmax - dmin
    L = dr
    N = (size[0] * size[1]) * n_channels
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    C3 = C2 / 2

    # Compute the mean
    mu_x = np.mean(img1)
    mu_y = np.mean(img2)

    # Compute the variance and covariance
    sigma_x = np.sqrt(np.sum(np.square(img1 - mu_x)) / (N - 1))
    sigma_y = np.sqrt(np.sum(np.square(img2 - mu_y)) / (N - 1))
    sigma_xy = np.sum((img1 - mu_x) * (img2 - mu_y)) / (N - 1)

    # Compute the luminance, contrast, and structure comparison
    l = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    c = (2 * sigma_x * sigma_y + C2) / (sigma_x ** 2 + sigma_y ** 2 + C2)
    s = (sigma_xy + C3) / (sigma_x * sigma_y + C3)

    ssim = l * c * s

    return ssim


def _chunk_points(points, chunk_size=1500):
    num_chunks = len(points) // chunk_size + (len(points) % chunk_size > 0)
    chunks = [points[i * chunk_size:(i + 1) * chunk_size]
              for i in range(num_chunks)]

    return chunks


def hausdorff_distance(img1, img2):
    points_x = np.transpose(np.nonzero(img1))
    points_y = np.transpose(np.nonzero(img2))

    # Create KDTrees
    kdtree_x = cKDTree(points_x)
    kdtree_y = cKDTree(points_y)

    # Divide points into chunks
    chunks_x = _chunk_points(points_x)
    chunks_y = _chunk_points(points_y)

    max_workers = max(1, os.cpu_count() // 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Compute the distance
        min_distances_x_futures = [executor.submit(
            kdtree_y.query, chunk) for chunk in chunks_x]
        min_distances_y_futures = [executor.submit(
            kdtree_x.query, chunk) for chunk in chunks_y]

        min_distances_x = [future.result()[0]
                           for future in min_distances_x_futures]
        min_distances_y = [future.result()[0]
                           for future in min_distances_y_futures]

    # Flatten the lists of minimum distances
    min_distances_x = [
        distance for chunk in min_distances_x for distance in chunk]
    min_distances_y = [
        distance for chunk in min_distances_y for distance in chunk]

    # Compute the maximum of the minimum distances
    d_x = max(min_distances_x)
    d_y = max(min_distances_y)

    return max(d_x, d_y)
