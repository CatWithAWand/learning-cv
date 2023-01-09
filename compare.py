import os
import time
import cv2
import numpy as np
import argparse
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
from skimage.metrics import structural_similarity

# This script can be used to compare two images based on the following metrics:
# MSE, MAE, PSNR, NCC, SSIM, and Hausdorff distance (if specified)
# The Hausdorff distance is computed using KDTrees in combination
# with chunking and multi-threading which makes it significantly fast.


def get_shape(img):
    shape = img.shape

    if len(shape) == 2:
        shape += (1,)

    return shape


def get_max_shape(img1, img2):
    shape1, shape2 = get_shape(img1), get_shape(img2)

    return (max(shape1[0], shape2[0]), max(shape1[1], shape2[1]), max(shape1[2], shape2[2]))


def get_min_shape(img1, img2):
    shape1, shape2 = get_shape(img1), get_shape(img2)

    return (min(shape1[0], shape2[0]), min(shape1[1], shape2[1]), min(shape1[2], shape2[2]))


def get_n_channels(img):
    return 1 if img.ndim == 2 else img.shape[2]


def get_dtype_range(dtype):
    return (np.iinfo(dtype).min, np.iinfo(dtype).max)


def pad_image(img, shape):
    img_shape = get_shape(img)

    if img_shape == shape:
        return img

    # Compute the amount of padding needed for each dimension
    padding = [(s - i) // 2 for s, i in zip(shape, img_shape)]

    # Pad the image using the calculated padding values
    return np.pad(img, [(p, s - i - p) for p, s, i in zip(padding, shape, img_shape)], mode='constant', constant_values=0)


def resize_image(img, shape):
    if get_shape(img) == shape:
        return img

    return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def MAE(img1, im2):
    # MAE = 1 / N * sum(|x - y|)
    abs_diff = np.abs(np.subtract(img1, img2, dtype=np.float64))
    mae = np.mean(abs_diff)

    return mae


def MSE(img1, img2):
    # MSE = 1 / N * sum((x - y) ** 2)
    diff = np.subtract(img1, img2, dtype=np.float64)
    mse = np.mean(np.square(diff))

    return mse


def PSNR(img1, img2):
    # PSNR = 10 * log10((max_i ** 2) / MSE)
    mse = MSE(img1, img2)

    if mse == 0:
        psnr = float('inf')
    else:
        max_i = get_dtype_range(img1.dtype.type)[1]
        psnr = 10 * np.log10(np.square(max_i) / mse)

    return psnr


# TODO: fix implementation, and implement Lewis numerator fft method
def NCC(img1, img2):
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
def SSIM(img1, img2):
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


def chunk_points(points, chunk_size=1500):
    num_chunks = len(points) // chunk_size + (len(points) % chunk_size > 0)
    chunks = [points[i * chunk_size:(i + 1) * chunk_size]
              for i in range(num_chunks)]

    return chunks


def hausdorff_distance(img1, img2):
    points_x = np.transpose(np.nonzero(img1))
    points_y = np.transpose(np.nonzero(img2))

    kdtree_x = cKDTree(points_x)
    kdtree_y = cKDTree(points_y)

    # Divide points into chunks
    chunks_x = chunk_points(points_x)
    chunks_y = chunk_points(points_y)

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


def image_similarity_score(mse, psnr, ssim):
    # Normalize the metrics within a range of 0-100
    # We invert MSE because we want a higher score for a lower MSE
    mse_norm = 100 - (100 * mse / (255 ** 2))
    psnr_norm = 100 * (100 if psnr > 100 else psnr) / 100
    ssim_norm = 100 * ssim

    # Weight the metrics
    mse_w = 0.2
    psnr_w = 0.3
    ssim_w = 0.5

    score = (mse_norm * mse_w) + (psnr_norm * psnr_w) + (ssim_norm * ssim_w)

    return score


def difference(img1, img2):
    max_i = get_dtype_range(img1.dtype.type)[1]
    diff = np.zeros_like(img1, dtype=np.float64)
    diff[img1 != img2] = max_i

    return diff


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--first', required=True,
                        help='path to the first image')
    parser.add_argument('-s', '--second', required=True,
                        help='path to the second image')
    parser.add_argument('-g', '--grayscale', action='store_true',
                        default=False, help='convert images to grayscale')
    parser.add_argument('-p', '--pad', action='store_true', default=False,
                        help='pad images to the same shape before comparing (pads with black, image is centered)')
    parser.add_argument('-r', '--resize', type=str, default=None,
                        help='resize images to the same shape before comparing (can be "upscale" or "downscale", uses bicubic interpolation)')
    parser.add_argument('-hd', '--hausdorff', action='store_true',
                        default=False, help='computes the Hausdorff distance between the images')
    parser.add_argument('-v', '--visualize', type=str, default=None,
                        help='output a visualization of the difference between the images to the specified directory')
    args = parser.parse_args()

    if args.pad and args.resize:
        print('Cannot use --pad and --resize at the same time.')
        exit()

    if args.resize and args.resize not in ['upscale', 'downscale']:
        print('Invalid value for --resize. Must be "upscale" or "downscale".')
        exit()

    if args.visualize:
        args.visualize = os.path.abspath(args.visualize)
        if not os.path.isdir(args.visualize):
            print(
                f'Invalid directory for --visualize: "{args.visualize}" is not a valid directory.')
            exit()

    color_flag = cv2.IMREAD_GRAYSCALE if args.grayscale else cv2.IMREAD_COLOR
    img1 = cv2.imread(args.first, color_flag)
    img2 = cv2.imread(args.second, color_flag)

    n_channels_1 = get_n_channels(img1)
    n_channels_2 = get_n_channels(img2)

    # If images have different channels, convert to grayscale
    if n_channels_1 != n_channels_1:
        print('Images have different number of color channels. Converting to grayscale.')
        if n_channels_1 > n_channels_1:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    shape1 = get_shape(img1)
    ratio1 = shape1[0] / shape1[1]
    shape2 = get_shape(img2)
    ratio2 = shape2[0] / shape2[1]

    # Check if images have different dimensions
    if shape1 != shape2 and not (args.pad or args.resize):
        print('Images have different dimensions. Use the --pad or --resize flags to compare images of different dimensions.')
        exit()

    if args.pad:
        target_shape = get_max_shape(img1, img2)
        print(f'Padding images to {target_shape[1]}x{target_shape[0]}')
        img1 = pad_image(img1, target_shape)
        img2 = pad_image(img2, target_shape)
    elif args.resize:
        target_shape = get_min_shape(
            img1, img2) if args.resize == 'downscale' else get_max_shape(img1, img2)
        print(f'Resizing images to {target_shape[1]}x{target_shape[0]}')
        img1 = resize_image(img1, target_shape)
        img2 = resize_image(img2, target_shape)

    mse = MSE(img1, img2)
    mae = MAE(img1, img2)
    psnr = PSNR(img1, img2)
    ncc = NCC(img1, img2)
    ssim = SSIM(img1, img2)
    # scikit_ssim = structural_similarity(
    #     img1, img2, multichannel=not args.grayscale)

    score = image_similarity_score(mse, psnr, ssim)

    print(f'Metrics:')
    print('- MSE: {}'.format(mse))
    print('- MAE: {}'.format(mae))
    print('- PSNR: {}'.format(psnr))
    print('- NCC: mean: {}, max: {}'.format(*ncc))
    print('- SSIM (self): {}'.format(ssim))
    # print('- SSIM (scikit): {}'.format(scikit_ssim))
    print('\nImage similarity score: {}'.format(score))

    if args.hausdorff:
        hausdorff_dist = hausdorff_distance(img1, img2)
        print('\nHausdorff distance: {}'.format(hausdorff_dist))

    if args.visualize:
        diff = difference(img1, img2)
        file_name1 = args.first.split('/')[-1].split('.')[0]
        file_name2 = args.second.split('/')[-1].split('.')[0]
        unix_timestamp = str(int(time.time()))
        file_path = os.path.join(
            args.visualize, f'diff_{file_name1}_{file_name2}_{unix_timestamp}.png')
        cv2.imwrite(file_path, diff)
