import os
import itertools
import cv2
import numpy as np
import argparse

from utils import is_supported_format, write_image


def random_crop(image, size, number):
    img_h = image.shape[0]
    img_w = image.shape[1]
    crops = []

    for _ in range(number):
        x = np.random.randint(0, img_w - size[0])
        y = np.random.randint(0, img_h - size[1])
        crops.append(image[y:y + size[1], x:x + size[0]])

    return crops


def coords_crop(image, coords):
    x, y, x2, y2 = coords

    crop = image[y:y2, x:x2]

    return crop


def get_grid(image, number):
    img_h = image.shape[0]
    img_w = image.shape[1]
    tile_h = img_h // number
    tile_w = img_w // number
    coords = []

    for (i, j) in itertools.product(range(number), range(number)):
        coords.append((j * tile_w, i * tile_h, (j + 1)
                      * tile_w, (i + 1) * tile_h))

    return coords


def get_grid_from_size(image, size):
    img_h = image.shape[0]
    img_w = image.shape[1]
    coords = []

    for (i, j) in itertools.product(range(0, img_h, size[1]), range(0, img_w, size[0])):
        coords.append((j, i, min(j + size[0], img_w), min(i + size[1], img_h)))

    return coords


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=None, required=True,
                        help='path to the input image or directory of images')
    parser.add_argument('-o', '--output', type=str, default=None,
                        required=True, help='path to the output image directory')
    parser.add_argument('-f', '--format', type=str, default='png',
                        help='output image format (png, jpg, etc.)')
    parser.add_argument('-g', '--grid', type=int, default=None,
                        help='crops image into a grid of n tiles')
    parser.add_argument('-gs', '--grid-from-size', action='store_true',
                        help='crops image into a grid based on the size of the crop')
    parser.add_argument('-r', '--random', type=int, default=None,
                        help='number of random crops to generate per image')
    parser.add_argument('-c', '--center', action='store_true',
                        help='crops image to center')
    parser.add_argument('-co', '--coords', nargs='+', type=int, default=None,
                        help='coordinates of the crop (x y x2 y2), eg. 0 0 256 256')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[256, 256],
                        help='size of the crop (width height), eg. 256 256')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'Invalid input path: "{args.input}" does not exist')
        exit()

    if not os.path.exists(args.output):
        print(f'Invalid output path: "{args.output}" does not exist')

    if args.format and not is_supported_format(args.format):
        print(f'Invalid output format: "{args.format}" is not supported')
        exit()

    # Get list of image file paths
    if os.path.isdir(args.input):
        images_filepaths = []
        for file in os.listdir(args.input):
            if is_supported_format(file):
                images_filepaths.append(os.path.join(args.input, file))
    else:
        if not is_supported_format(args.input):
            print(f'Invalid input format: "{args.input}" is not supported')
            exit()

        images_filepaths = [args.input]

    # Load images
    images = {filepath: cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
              for filepath in images_filepaths}

    for filepath, image in images.items():
        file_name = os.path.splitext(os.path.basename(filepath))[0]

        if args.grid or args.grid_from_size:
            coords = get_grid(image, args.grid) if args.grid else get_grid_from_size(
                image, args.size)
            for (x, y, x2, y2) in coords:
                crop = coords_crop(image, [x, y, x2, y2])
                write_image(crop, args.output,
                            prefix=f'grid_crop_{file_name}', suffix=f'{x}_{y}_{x2}_{y2}', ext=args.format)

        if args.random:
            crops = random_crop(image, args.size, args.random)
            for i, crop in enumerate(crops):
                write_image(crop, args.output,
                            prefix=f'random_crop_{file_name}', suffix=f'{i}', ext=args.format)

        if args.center:
            x = image.shape[1] // 2 - args.size[0] // 2
            y = image.shape[0] // 2 - args.size[1] // 2
            x2 = x + args.size[0]
            y2 = y + args.size[1]
            crop = coords_crop(image, [x, y, x2, y2])
            write_image(crop, args.output,
                        prefix=f'center_crop_{file_name}', ext=args.format)

        if args.coords:
            if len(args.coords) != 4:
                print('Invalid coordinates: must be 4 values [x, y, x2, y2]')
                exit()

            if args.coords[0] >= args.coords[2] or args.coords[1] >= args.coords[3]:
                print(
                    'Invalid coordinates: x must be less than x2, and y must be less than y2')
                exit()

            crop = coords_crop(image, args.coords)
            write_image(crop, args.output,
                        prefix=f'coords_crop_{file_name}', ext=args.format)