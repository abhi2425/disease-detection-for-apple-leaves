import pickle
from app import bgr_hsv, fd_haralick, fd_histogram, fd_hu_moments, img_segmentation, rgb_bgr
import cv2
import numpy as np
import matplotlib.pyplot as plt

loaded_model = pickle.load(open("models/{}.pkl".format("rf"), "rb"))

image = cv2.imread("dataset/test/Apple___Cedar_apple_rust/fa148283-75ee-4806-8282-63662a2e05c1___FREC_C.Rust 4416.JPG")
print(image.shape)
print(loaded_model)
rgb_image = rgb_bgr(image)
hsv_image = bgr_hsv(rgb_image)
segmented_image = img_segmentation(rgb_image, hsv_image)
hu_moments = fd_hu_moments(segmented_image)
haralick_texture = fd_haralick(segmented_image)
color_histogram = fd_histogram(segmented_image)
global_feature = np.hstack([color_histogram, haralick_texture, hu_moments])

print(global_feature)

prediction = loaded_model.predict([global_feature])
print(prediction)