# Create a Streamlit web app for surface crack detection using a trained model

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd

from src.model import SimpleCNN
st.set_page_config(layout="wide")
st.title("Surface Crack Detection")
# wide layout
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor()
])

def segment_image(image, threshold=125):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def load_model(model_path: str):
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, image: np.ndarray):
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    return output

def main():
    model_path = "models/model.pth"
    model = load_model(model_path)

    c1, c2 = st.columns([1, 1])
    with c1:
        uploaded_image = st.file_uploader("Choose an image...", type="jpg")
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            segmented_image = segment_image(cv2_image)
            st.image(image, caption="Uploaded Image.", use_container_width=True)

            if st.button("Predict"):
                image_tensor = image_transform(image)
                output = predict(model, image_tensor)
                output = torch.sigmoid(output).cpu().numpy()
                prediction = "Crack" if output[0] > 0.5 else "No Crack"
                with c2:
                    st.success(f"Prediction: {prediction}")

                    # Display the bar chart showing probability of No crack (0) and Crack (1)
                    probability_df = pd.DataFrame({"Label": ["No Crack", "Crack"], "Probability": [1 - output[0][0], output[0][0]]})

                    st.bar_chart(probability_df.set_index("Label"))

                    # Display the segmented image
                    st.image(segmented_image, caption="Segmented Image", use_container_width=True)



if __name__ == "__main__":
    main()