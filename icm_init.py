import cv2
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from numba import njit


@njit
def stretching(image, stretch_max_val, stretch_min_val):
    row = len(image)
    column = len(image[0])

    for color_channel in range(0, 3):
        channel_max_value = np.max(image[:, :, color_channel])
        channel_min_value = np.min(image[:, :, color_channel])
        for row_val in range(row):
            for col_val in range(column):
                image[row_val, col_val, color_channel] = (image[row_val, col_val, color_channel] - channel_min_value) * (stretch_max_val - stretch_min_val) / (channel_max_value - channel_min_value) + stretch_min_val

    return image


@njit
def global_stretching(channel, row, column):
    channel_min_value = np.min(channel)
    channel_max_value = np.max(channel)
    stretched_channel = np.zeros((row, column))
    for row_val in range(row):
        for col_val in range(column):
            output_pixel_value = (channel[row_val][col_val] - channel_min_value) * (1 / (channel_max_value - channel_min_value))
            stretched_channel[row_val][col_val] = output_pixel_value

    return stretched_channel


def sv_stretching(radiance):
    row = len(radiance)
    column = len(radiance[0])

    image_hsv = rgb2hsv(radiance)
    hue, saturation, value = cv2.split(image_hsv)

    saturation_channel_stretching = global_stretching(saturation, row, column)
    value_channel_stretching = global_stretching(value, row, column)

    sv_stretched_image = np.zeros((row, column, 3), np.float64)

    sv_stretched_image[:, :, 0] = hue
    sv_stretched_image[:, :, 1] = saturation_channel_stretching
    sv_stretched_image[:, :, 2] = value_channel_stretching

    rgb_image = hsv2rgb(sv_stretched_image) * 255

    return rgb_image


def radiance_rgb(radiance):
    radiance = np.clip(radiance, 0, 255)
    radiance = np.uint8(radiance)

    return radiance
