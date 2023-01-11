import sys
import os
import cv2
import numpy as np
import argparse
# from skimage.metrics import structural_similarity

from metrics import (
    mean_absolute_error,
    mean_squared_error,
    peak_signal_noise_ratio,
    normalized_cross_correlation,
    structural_similarity_index,
    hausdorff_distance,
)
from utils import (
    is_supported_format,
    get_shape,
    get_max_shape,
    get_min_shape,
    get_n_channels,
    get_dtype_range,
    pad_image,
    resize_image,
    write_image,
)

# This script can be used to compare two images based on the following metrics:
# MSE, MAE, PSNR, NCC, SSIM, and Hausdorff distance (if specified)
# The Hausdorff distance is computed using KDTrees in combination
# with chunking and multi-threading which makes it significantly fast.


def image_similarity_score(mse, psnr, ssim):
    # Normalize the metrics within a range of 0-100
    # We invert MSE because we want a higher score for a lower MSE
    mse_norm = 100 - (100 * mse / (255**2))
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


def main(args):
    if not os.path.isfile(args.first) or not is_supported_format(args.first):
        print(f'Invalid first image: "{args.first}" is not a valid file.')
        sys.exit()

    if not os.path.isfile(args.second) or not is_supported_format(args.second):
        print(f'Invalid second image: "{args.second}" is not a valid file.')
        sys.exit()

    if args.pad and args.resize:
        print('Cannot use --pad and --resize at the same time.')
        sys.exit()

    if args.resize and args.resize not in ['upscale', 'downscale']:
        print('Invalid value for --resize. Must be "upscale" or "downscale".')
        sys.exit()

    if args.visualize:
        args.visualize = os.path.abspath(args.visualize)
        if not os.path.isdir(args.visualize):
            print(
                'Invalid directory for --visualize: ' \
                f'"{args.visualize}" is not a valid directory.'
            )
            sys.exit()

    print(f'Comparing images {args.first} and {args.second}')

    color_flag = cv2.IMREAD_GRAYSCALE if args.grayscale else cv2.IMREAD_COLOR
    img1 = cv2.imread(args.first, color_flag, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(args.second, color_flag, cv2.IMREAD_UNCHANGED)

    n_channels_1 = get_n_channels(img1)
    n_channels_2 = get_n_channels(img2)

    # If images have different channels, convert to grayscale
    if n_channels_1 != n_channels_2:
        print(
            'Images have different number of color channels. ' \
            'Converting to grayscale...'
        )
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    shape1 = get_shape(img1)
    shape2 = get_shape(img2)

    # Check if images have different dimensions
    if shape1 != shape2 and not (args.pad or args.resize):
        print(
            'Images have different dimensions. Use the --pad or ' \
            '--resize flags to compare images of different dimensions.'
        )
        sys.exit()

    if args.pad:
        target_shape = get_max_shape(img1, img2)
        print(f'Padding images to {target_shape[1]}x{target_shape[0]}...')
        img1 = pad_image(img1, target_shape)
        img2 = pad_image(img2, target_shape)
    elif args.resize:
        target_shape = get_min_shape(
            img1, img2) if args.resize == 'downscale' else get_max_shape(
                img1, img2)
        print(f'Resizing images to {target_shape[1]}x{target_shape[0]}...')
        img1 = resize_image(img1, target_shape)
        img2 = resize_image(img2, target_shape)

    mae = mean_absolute_error(img1, img2)
    mse = mean_squared_error(img1, img2)
    psnr = peak_signal_noise_ratio(img1, img2)
    ncc = normalized_cross_correlation(img1, img2)
    ssim = structural_similarity_index(img1, img2)
    # scikit_ssim = structural_similarity(
    #     img1, img2, multichannel=not args.grayscale)

    score = image_similarity_score(mse, psnr, ssim)

    print('Metrics:')
    print(f'- MAE: {mae}')
    print(f'- MSE: {mse}')
    print(f'- PSNR: {psnr}')
    print(f'- NCC: {ncc}')
    print(f'- SSIM: {ssim}')
    # print(f'- SSIM (scikit): {scikit_ssim}')
    print(f'\nImage similarity score: {score}')

    if args.hausdorff:
        hausdorff_dist = hausdorff_distance(img1, img2)
        print(f'\nHausdorff distance: {hausdorff_dist}')

    if args.visualize:
        diff = difference(img1, img2)
        file_name1 = os.path.splitext(os.path.basename(args.first))[0]
        file_name2 = os.path.splitext(os.path.basename(args.second))[0]
        write_image(diff,
                    args.visualize,
                    prefix=f'diff_{file_name1}_{file_name2}',
                    ext='png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
                        '--first',
                        required=True,
                        help='path to the first image')
    parser.add_argument('-s',
                        '--second',
                        required=True,
                        help='path to the second image')
    parser.add_argument('-g',
                        '--grayscale',
                        action='store_true',
                        default=False,
                        help='convert images to grayscale')
    parser.add_argument(
        '-p',
        '--pad',
        action='store_true',
        default=False,
        help='pad images to the same shape before comparing ' \
        '(pads with black, image is centered)')
    parser.add_argument(
        '-r',
        '--resize',
        type=str,
        default=None,
        help=
        'resize images to the same shape before comparing (can be "upscale" ' \
            'or "downscale", uses bicubic interpolation)')
    parser.add_argument(
        '-hd',
        '--hausdorff',
        action='store_true',
        default=False,
        help='computes the Hausdorff distance between the images')
    parser.add_argument(
        '-v',
        '--visualize',
        type=str,
        default=None,
        help=
        'output a visualization of the difference between the images ' \
            'to the specified directory'
    )

    main(parser.parse_args())
