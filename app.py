import os
import cv2
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mahotas
import pickle


# Set seed for reproducibility
seed = 9

# Models dictionary
models = {
    "Logistic Regression": "lr",
    "K Neighbors Classifier": "knn",
    "Decision Tree Classifier": "dtc",
    "Random Forest Classifier": "rf",
    "Linear Discriminant Analysis": "lda",
    "Gaussian Naive Bayes": "nb",
    "SVC": "svm",
}

# Model labels
labels_mapping = {
    0: "Apple___Apple_scab",
    1: "Apple___Black_rot",
    2: "Apple___Cedar_apple_rust",
    3: "Apple___healthy"
}

# Helper functions for feature extraction

def rgb_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def bgr_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def img_segmentation(rgb_img, hsv_img):
    lower_green = np.array([25, 0, 20])
    upper_green = np.array([100, 255, 255])
    healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    result = cv2.bitwise_and(rgb_img, rgb_img, mask=healthy_mask)
    lower_brown = np.array([10, 0, 10])
    upper_brown = np.array([30, 255, 255])
    disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
    disease_result = cv2.bitwise_and(rgb_img, rgb_img, mask=disease_mask)
    final_mask = healthy_mask + disease_mask
    final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
    return final_result

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bins = 8
    hist = cv2.calcHist(
        [image], [0, 1, 2], mask, [bins, bins, bins], [0, 256, 0, 256, 0, 256]
    )
    cv2.normalize(hist, hist)
    return hist.flatten()

def main():
    st.set_page_config(
        page_title='Plant Disease Detection for Apple Leaves',
        page_icon="🌿"                  
        )
    st.title("Plant Disease Detection for Apple Leaves")
    st.subheader("Upload an image and select a model to detect the disease")

    uploaded_file = st.file_uploader("Upload an image of an apple leaf", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        model_name =  st.selectbox("Select a model for disease detection", list(models.keys()))

        if st.button("Detect"):
          
            # Convert uploaded image to RGB and display
            rgb_image = rgb_bgr(image)
            st.subheader("RGB Image")
            fig_rgb = plt.figure()
            plt.imshow(rgb_image)
            plt.axis('off')
            st.pyplot(fig_rgb)

            # Convert RGB image to HSL and display
            hsl_image = bgr_hsv(rgb_image)
            st.subheader("HSL Image")
            fig_hsl = plt.figure()
            plt.imshow(hsl_image)
            plt.axis('off')
            st.pyplot(fig_hsl)

            # Perform image segmentation
            segmented_image = img_segmentation(rgb_image, hsl_image)

            # Display segmented images
            st.subheader("Segmented Images")
            fig_segmented = plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(segmented_image)
            plt.axis('off')
            plt.title("Segmented Image")

            plt.subplot(1, 3, 2)
            lower_green = np.array([25, 0, 20])
            upper_green = np.array([100, 255, 255])
            mask = cv2.inRange(hsl_image, lower_green, upper_green)
            result = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)
            plt.imshow(mask, cmap="gray")
            plt.axis('off')
            plt.title("Healthy Mask")

            plt.subplot(1, 3, 3)
            lower_brown = np.array([10, 0, 10])
            upper_brown = np.array([30, 255, 255])
            disease_mask = cv2.inRange(hsl_image, lower_brown, upper_brown)
            disease_result = cv2.bitwise_and(rgb_image, rgb_image, mask=disease_mask)
            plt.imshow(disease_mask)
            plt.axis('off')
            plt.title("Disease Mask")

            st.pyplot(fig_segmented)

            # Calculate feature descriptors
            hu_moments = fd_hu_moments(segmented_image)
            haralick_texture = fd_haralick(segmented_image)
            color_histogram = fd_histogram(segmented_image)
            
            # Concatenate global features
            global_features = np.hstack([color_histogram, haralick_texture, hu_moments])

            # Load the selected model
            model = models[model_name]
            loaded_model = pickle.load(open("models/{}.pkl".format(model), "rb"))

            # Predict the disease label
            prediction = loaded_model.predict([global_features])
            st.subheader("Prediction")
            st.write("Prediction: ", prediction)
            
            predicted_label = labels_mapping[prediction[0]]
            # accuracy = loaded_model.score([global_features], [prediction])

            st.subheader("Disease Prediction")
            st.write("Predicted Label-: ", predicted_label)

if __name__ == '__main__':
    main()