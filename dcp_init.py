import cv2
import math
import numpy as np
from numba import njit, prange


def dark_channel(image, patch_size):
    blue, green, red = cv2.split(image)
    darkest_channel = cv2.min(cv2.min(red, green), blue)
    patch = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(darkest_channel, patch)
    return dark


def atmosphere_radiance(image, dark):
    row, column = image.shape[:2]
    image_size = row * column
    numpx = int(max(math.floor(image_size / 1000), 1))
    dark_vector = dark.reshape(image_size, 1)
    image_vector = image.reshape(image_size, 3)

    dark_values = dark_vector.argsort()
    dark_values = dark_values[image_size - numpx::]

    atmosphere_sum = np.zeros((1, 3))
    for i in range(1, numpx):
        atmosphere_sum = atmosphere_sum + image_vector[dark_values[i]]

    atmospheric_radiance = atmosphere_sum / numpx

    return atmospheric_radiance


@njit(parallel=True)
def transmission_estimate(image, atmospheric_radiance):
    revised_image = np.empty(image.shape, image.dtype)

    for channel in prange(0, 3):
        revised_image[:, :, channel] = image[:, :, channel] / atmospheric_radiance[0, channel]

    return revised_image


def transmission_estimate_result(revised_image, omega, patch_size):
    transmission = 1 - omega * dark_channel(revised_image, patch_size)

    return transmission


def guided_filter(image, guidance, radius, epsilon):
    mean_image = cv2.boxFilter(image, cv2.CV_64F, (radius, radius))
    mean_guidance = cv2.boxFilter(guidance, cv2.CV_64F, (radius, radius))
    image_guidance_correlation = cv2.boxFilter(image * guidance, cv2.CV_64F, (radius, radius))
    image_correlation = cv2.boxFilter(image * image, cv2.CV_64F, (radius, radius))

    variance_image = image_correlation - mean_image ** 2
    covariance_image_with_tx = image_guidance_correlation - mean_image * mean_guidance

    a = covariance_image_with_tx / (variance_image + epsilon)
    b = mean_guidance - a * mean_image

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))

    guided_filtered_image = mean_a * image + mean_b

    return guided_filtered_image


def transmission_refine(image, transmission_estimate, radius, epsilon):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.float64(gray_image) / 255

    transmission_refined = guided_filter(gray_image, transmission_estimate, radius, epsilon)

    return transmission_refined


def recover(image, transmission, atmospheric_radiance, tx):
    image_recovered = np.empty(image.shape, image.dtype)
    transmission = cv2.max(transmission, tx)

    for channel in range(0, 3):
        image_recovered[:, :, channel] = (image[:, :, channel] - atmospheric_radiance[0, channel]) / transmission + atmospheric_radiance[0, channel]

    return image_recovered


def dcp(image, patch_size=15, omega=0.95, radius=60, epsilon=0.0001, tx=0.1):
    intensity = image.astype('float64') / 255
    dark = dark_channel(intensity, patch_size)
    atmospheric_radiance = atmosphere_radiance(intensity, dark)
    estimated_transmission = transmission_estimate(intensity, atmospheric_radiance)
    transmission_result = transmission_estimate_result(estimated_transmission, omega, patch_size)
    refined_transmission = transmission_refine(image, transmission_result, radius, epsilon)
    scene_radiance = recover(intensity, refined_transmission, atmospheric_radiance, tx)

    return scene_radiance
