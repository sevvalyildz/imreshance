import numpy as np
from guided_filter import GuidedFilter
from numba import njit, prange


class Node:
    def __init__(self, row, column, value):
        self.row = row
        self.column = column
        self.value = value


def get_atmospheric_light(image, dark_channel):
    row = dark_channel.shape[0]
    column = dark_channel.shape[1]
    nodes = []

    for row in range(row):
        for col in range(column):
            nodes.append(Node(row, col, dark_channel[row, col]))

    nodes = sorted(nodes, key=lambda node: node.value, reverse=True)

    atmospheric_light = image[nodes[0].row, nodes[0].column, :]

    return atmospheric_light


@njit(parallel=True)
def get_min_channel(image):
    gray_image = np.zeros((image.shape[0], image.shape[1]))

    for row in prange(image.shape[0]):
        for col in prange(image.shape[1]):
            local_min = 255

            for ch in prange(2):
                if image.item((row, col, ch)) < local_min:
                    local_min = image[row, col, ch]

            gray_image[row, col] = local_min

    return gray_image


@njit(parallel=True)
def get_dark_channel(image, patch_size):
    add_size = int((patch_size - 1) / 2)
    new_row = image.shape[0] + patch_size - 1
    new_column = image.shape[1] + patch_size - 1

    image_middle = np.zeros((new_row, new_column))
    image_middle[:, :] = 255
    image_middle[add_size:new_row - add_size, add_size:new_column - add_size] = image
    image_dark = np.zeros((image.shape[0], image.shape[1]), np.uint8)

    for mid_row in range(add_size, new_row - add_size):
        for mid_col in range(add_size, new_column - add_size):
            local_min = 255

            for row in range(mid_row - add_size, mid_row + add_size + 1):
                for col in range(mid_col - add_size, mid_col + add_size + 1):
                    if image_middle[mid_row, mid_col] < local_min:
                        local_min = image_middle[mid_row, mid_col]

            image_dark[mid_row - add_size, mid_col - add_size] = local_min

    return image_dark


@njit
def get_normalization_channel(image, atmospheric_light):
    image_normalization = np.zeros((image.shape[0], image.shape[1]))

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            local_min = 1

            for ch in range(2):
                image_normal = image.item((row, col, ch)) / atmospheric_light[ch]

                if image_normal < local_min:
                    local_min = image_normal

            image_normalization[row, col] = local_min

    return image_normalization


@njit
def get_transmission(norm_channel_image, patch_size):
    add_size = int((patch_size - 1) / 2)
    new_row = norm_channel_image.shape[0] + patch_size - 1
    new_col = norm_channel_image.shape[1] + patch_size - 1

    image_middle = np.zeros((new_row, new_col))
    image_middle[:, :] = 1
    image_middle[add_size:new_row - add_size, add_size:new_col - add_size] = norm_channel_image
    image_dark = np.zeros((norm_channel_image.shape[0], norm_channel_image.shape[1]))

    for mid_row in range(add_size, new_row - add_size):
        for mid_col in range(add_size, new_col - add_size):
            local_min = 1

            for row in range(mid_row - add_size, mid_row + add_size + 1):
                for col in range(mid_col - add_size, mid_col + add_size + 1):
                    if image_middle[row, col] < local_min:
                        local_min = image_middle[row, col]

            image_dark[mid_row - add_size, mid_col - add_size] = local_min

    transmission = 1 - image_dark

    return transmission


def transmission_clip(transmission):
    clipped_transmission = np.clip(transmission, 0.1, 0.9)
    return clipped_transmission


def transmission_refine(guidance, transmission, radius, epsilon):
    filter = GuidedFilter(guidance, radius, epsilon)
    refined_transmission = filter.guided_filter(transmission)
    refined_transmission = np.clip(refined_transmission, 0.1, 0.9)

    return refined_transmission


def scene_radiance(image, transmission, atmospheric_light):
    atmospheric_light = np.array(atmospheric_light)
    image = np.float64(image)
    radiance = np.zeros(image.shape)
    transmission = np.clip(transmission, 0.2, 0.9)

    for ch in range(3):
        radiance[:, :, ch] = (image[:, :, ch] - atmospheric_light[ch]) / transmission + atmospheric_light[ch]

    radiance = np.clip(radiance, 0, 255)
    radiance = np.uint8(radiance)

    return radiance


def udcp(image, patch_size, radius, epsilon):
    min_channel_image = get_min_channel(image)
    gb_dark_channel = get_dark_channel(min_channel_image, patch_size)
    atmospheric_light = get_atmospheric_light(image, gb_dark_channel)
    norm_channel_image = get_normalization_channel(image, atmospheric_light)
    transmission = get_transmission(norm_channel_image, udcpVal.patch_size)
    clipped = transmission_clip(transmission)
    refined_transmission = transmission_refine(image, clipped, radius, epsilon)
    radiance = scene_radiance(image, refined_transmission, atmospheric_light)

    return radiance
