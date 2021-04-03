import numpy as np
import cv2
import scipy.ndimage as ndi


def getPSNR(I1, I2):
    """
    Higher - better
    """
    s1 = cv2.absdiff(I1, I2) #|I1 - I2|
    s1 = np.float32(s1)     # cannot make a square on 8 bits
    s1 = s1 * s1            # |I1 - I2|^2
    sse = s1.sum()          # sum elements per channel
    if sse <= 1e-10:        # sum channels
        print("The same image")            # for small values return zero
        return 0
    else:
        shape = I1.shape
        mse = 1.0 * sse / (shape[0])
        psnr = 10.0 * np.log10((255 * 255) / mse)
        print(f"Signal to noise ratio (higher - better): {psnr: .2f}")
        return psnr


def remove_salt_pepper(image, ksize=5):
    new_image = cv2.medianBlur(image, ksize)
    getPSNR(image, new_image)
    return new_image


def remove_gausse_h1(image):
    H_1 = np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]])
    H_1_coef = 1 / 16

    image_gausse_H1 = ndi.convolve(image, H_1) * H_1_coef
    getPSNR(image, image_gausse_H1)
    return image_gausse_H1


def remove_gausse_h2(image):
    H_2 = np.array([[2, 1, 2],
                    [1, 2, 1],
                    [2, 1, 2]])
    H_2_coef = 1 / 14

    image_gausse_H2 = ndi.convolve(image, H_2) * H_2_coef
    getPSNR(image, image_gausse_H2)

    return image_gausse_H2


def remove_gausse_median_filter(image, win_size=7):
    image_noised_g_median = ndi.median_filter(image, win_size)
    getPSNR(image, image_noised_g_median)

    return image_noised_g_median


def make_binary(image, method=cv2.THRESH_OTSU):
    out_binary = cv2.threshold(image, 0, 255, method)[1]
    getPSNR(image, out_binary)
    return out_binary
