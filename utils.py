import os
import time
import cv2
import numpy as np

SUPPORT_IMAGE_FORMATS = [
    'bmp', 'dib', 'jpeg', 'jpg', 'jpe', 'jp2', 'png', 'webp', 'pbm', 'pgm',
    'ppm', 'pxm', 'pnm', 'pfm', 'sr', 'ras', 'tiff', 'tif', 'exr', 'hdr', 'pic'
]


def is_supported_format(filepath):
    ext = filepath if '.' not in filepath else os.path.splitext(
        filepath)[-1].lower()
    ext = ext[1:] if ext.startswith('.') else ext

    return ext in SUPPORT_IMAGE_FORMATS


def get_shape(img):
    shape = img.shape

    if len(shape) == 2:
        shape += (1, )

    return shape


def get_max_shape(img1, img2):
    shape1, shape2 = get_shape(img1), get_shape(img2)

    return (max(shape1[0],
                shape2[0]), max(shape1[1],
                                shape2[1]), max(shape1[2], shape2[2]))


def get_min_shape(img1, img2):
    shape1, shape2 = get_shape(img1), get_shape(img2)

    return (min(shape1[0],
                shape2[0]), min(shape1[1],
                                shape2[1]), min(shape1[2], shape2[2]))


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
    return np.pad(img, [(p, s - i - p)
                        for p, s, i in zip(padding, shape, img_shape)],
                  mode='constant',
                  constant_values=0)


def resize_image(img, shape):
    if get_shape(img) == shape:
        return img

    return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def write_image(img, output_dir, prefix='', suffix='', ext='png'):
    if not os.path.exists(output_dir):
        raise Exception(
            f'Invalid output directory: "{output_dir}" does not exist.')

    if ext in ['jpeg', 'jpg', 'jpe', 'jp2']:
        qflag = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    elif ext == 'png':
        qflag = [int(cv2.IMWRITE_PNG_COMPRESSION), 5]
    else:
        qflag = None

    unix_timestamp = str(int(time.time()))
    file_path = os.path.join(
        output_dir,
        f'{prefix}{"_" if prefix else ""}{unix_timestamp}' \
        f'{"_" if suffix else ""}{suffix}.{ext}'
    )
    cv2.imwrite(file_path, img, qflag)
