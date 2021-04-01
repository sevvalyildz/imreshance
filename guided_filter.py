import cv2
import numpy as np


def dtype_conv_float32(image):
    if image.dtype == np.float32:
        return image
    return (1.0 / 255.0) * np.float32(image)


class GuidedFilter:
    def __init__(self, guidance, radius, epsilon):
        self.guidance = dtype_conv_float32(guidance)
        self.radius = 2 * radius + 1
        self.epsilon = epsilon

        self.guidance_blue = self.guidance[:, :, 0]
        self.guidance_green = self.guidance[:, :, 1]
        self.guidance_red = self.guidance[:, :, 2]

        self.guidance_blue_mean = 0
        self.guidance_green_mean = 0
        self.guidance_red_mean = 0

        self.guidance_blue_blue_inverse = 0
        self.guidance_green_green_inverse = 0
        self.guidance_green_blue_inverse = 0
        self.guidance_red_red_inverse = 0
        self.guidance_red_green_inverse = 0
        self.guidance_red_blue_inverse = 0

    def init_filter(self):
        self.guidance_blue_mean = cv2.blur(self.guidance_blue, (self.radius, self.radius))
        self.guidance_green_mean = cv2.blur(self.guidance_green, (self.radius, self.radius))
        self.guidance_red_mean = cv2.blur(self.guidance_red, (self.radius, self.radius))

        guidance_blue_blue_variance = cv2.blur(self.guidance_blue ** 2, (self.radius, self.radius)) - self.guidance_blue_mean ** 2 + self.epsilon
        guidance_green_green_variance = cv2.blur(self.guidance_green ** 2, (self.radius, self.radius)) - self.guidance_green_mean ** 2 + self.epsilon
        guidance_green_blue_variance = cv2.blur(self.guidance_green * self.guidance_blue, (self.radius, self.radius)) - self.guidance_green_mean * self.guidance_blue_mean
        guidance_red_red_variance = cv2.blur(self.guidance_red ** 2, (self.radius, self.radius)) - self.guidance_red_mean ** 2 + self.epsilon
        guidance_red_green_variance = cv2.blur(self.guidance_red * self.guidance_green, (self.radius, self.radius)) - self.guidance_red_mean * self.guidance_green_mean
        guidance_red_blue_variance = cv2.blur(self.guidance_red * self.guidance_blue, (self.radius, self.radius)) - self.guidance_red_mean * self.guidance_blue_mean

        guidance_blue_blue_inverse = guidance_red_red_variance * guidance_green_green_variance - guidance_red_green_variance ** 2
        guidance_green_green_inverse = guidance_red_red_variance * guidance_blue_blue_variance - guidance_red_blue_variance ** 2
        guidance_green_blue_inverse = guidance_red_blue_variance * guidance_red_green_variance - guidance_red_red_variance * guidance_green_blue_variance
        guidance_red_red_inverse = guidance_green_green_variance * guidance_blue_blue_variance - guidance_green_blue_variance ** 2
        guidance_red_green_inverse = guidance_red_green_variance * guidance_red_blue_variance - guidance_red_green_variance * guidance_blue_blue_variance
        guidance_red_blue_inverse = guidance_red_green_variance * guidance_green_blue_variance - guidance_green_green_variance * guidance_red_blue_variance

        guidance_covariance = guidance_red_red_inverse * guidance_red_red_variance + guidance_red_green_inverse * guidance_red_green_variance + guidance_red_blue_inverse * guidance_red_blue_variance

        guidance_blue_blue_inverse /= guidance_covariance
        guidance_green_green_inverse /= guidance_covariance
        guidance_green_blue_inverse /= guidance_covariance
        guidance_red_red_inverse /= guidance_covariance
        guidance_red_green_inverse /= guidance_covariance
        guidance_red_blue_inverse /= guidance_covariance

        self.guidance_blue_blue_inverse = guidance_blue_blue_inverse
        self.guidance_green_green_inverse = guidance_green_green_inverse
        self.guidance_green_blue_inverse = guidance_green_blue_inverse
        self.guidance_red_red_inverse = guidance_red_red_inverse
        self.guidance_red_green_inverse = guidance_red_green_inverse
        self.guidance_red_blue_inverse = guidance_red_blue_inverse

    def guided_filter(self, input_image):
        input_image_mean = cv2.blur(input_image, (self.radius, self.radius))
        input_blue_mean = cv2.blur(self.guidance_blue * input_image, (self.radius, self.radius))
        input_green_mean = cv2.blur(self.guidance_green * input_image, (self.radius, self.radius))
        input_red_mean = cv2.blur(self.guidance_red * input_image, (self.radius, self.radius))

        input_blue_covariance = input_blue_mean - self.guidance_blue_mean * input_image_mean
        input_green_covariance = input_green_mean - self.guidance_green_mean * input_image_mean
        input_red_covariance = input_red_mean - self.guidance_red_mean * input_image_mean

        a_blue = self.guidance_red_blue_inverse * input_red_covariance + self.guidance_green_blue_inverse * input_green_covariance + self.guidance_blue_blue_inverse * input_red_covariance
        a_green = self.guidance_red_green_inverse * input_red_covariance + self.guidance_green_green_inverse * input_green_covariance + self.guidance_green_blue_inverse * input_blue_covariance
        a_red = self.guidance_red_red_inverse * input_red_covariance + self.guidance_red_green_inverse * input_green_covariance + self.guidance_red_blue_inverse * input_blue_covariance

        b = input_image_mean - a_red * self.guidance_red_mean - a_green * self.guidance_green_mean - a_blue * self.guidance_blue_mean

        a_blue_mean = cv2.blur(a_blue, (self.radius, self.radius))
        a_green_mean = cv2.blur(a_green, (self.radius, self.radius))
        a_red_mean = cv2.blur(a_red, (self.radius, self.radius))
        b_mean = cv2.blur(b, (self.radius, self.radius))

        output_image = a_red_mean * self.guidance_red + a_green_mean * self.guidance_green + a_blue_mean * self.guidance_blue + b_mean
        return output_image