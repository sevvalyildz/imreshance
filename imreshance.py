import icm_init, dcp_init, udcp_init


def icm(image, stretch_max_val=255, stretch_min_val=0):
    image = icm_init.stretching(image, stretch_max_val, stretch_min_val)
    radiance = icm_init.radiance_rgb(image)
    radiance = icm_init.sv_stretching(radiance)
    radiance = icm_init.radiance_rgb(radiance)

    return radiance


def dcp(image, patch_size=15, omega=0.95, radius=60, epsilon=0.0001, tx=0.1):
    intensity = image.astype('float64') / 255
    dark = dcp_init.dark_channel(intensity, patch_size)
    atmospheric_radiance = dcp_init.atmosphere_radiance(intensity, dark)
    estimated_transmission = dcp_init.transmission_estimate(intensity, atmospheric_radiance)
    transmission_result = dcp_init.transmission_estimate_result(estimated_transmission, omega, patch_size)
    refined_transmission = dcp_init.transmission_refine(image, transmission_result, radius, epsilon)
    scene_radiance = dcp_init.recover(intensity, refined_transmission, atmospheric_radiance, tx)

    return scene_radiance


def udcp(image, patch_size=9, radius=50, epsilon=0.001):
    min_channel_image = udcp_init.get_min_channel(image)
    gb_dark_channel = udcp_init.get_dark_channel(min_channel_image, patch_size)
    atmospheric_light = udcp_init.get_atmospheric_light(image, gb_dark_channel)
    norm_channel_image = udcp_init.get_normalization_channel(image, atmospheric_light)
    transmission = udcp_init.get_transmission(norm_channel_image, patch_size)
    clipped = udcp_init.transmission_clip(transmission)
    refined_transmission = udcp_init.transmission_refine(image, clipped, radius, epsilon)
    radiance = udcp_init.scene_radiance(image, refined_transmission, atmospheric_light)

    return radiance
