import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from torchvision import models  # Import this for efficientnet_b0
import matplotlib.pyplot as plt


# Define the PlantDiseaseClassifier model
class PlantDiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseClassifier, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=False)
        num_ftrs = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


# Load the trained model
def load_model(model_path, num_classes):
    model = PlantDiseaseClassifier(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


# Define the transformation for input images
def get_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


# Function to predict the disease and confidence
def predict_disease(model, image, class_names, transform):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return class_names[predicted.item()], confidence.item()


# Streamlit front end
def main():
    st.title("Plant Disease Detection")
    st.write("Upload an image of a plant leaf to detect the disease and the confidence level.")

    # Upload image
    uploaded_file = st.file_uploader("Choose a leaf image...", type="jpg")

    # Class names
    class_names = [
        "Corn - Common Rust", "Corn - Gray Leaf Spot", "Corn - Northern Leaf Blight", "Corn - Healthy",
        "Potato - Early Blight", "Potato - Late Blight", "Potato - Healthy", "Rice - Brown Spot",
        "Rice - Leaf Blast", "Rice - Neck Blast", "Rice - Healthy", "Sugarcane - Bacterial Blight",
        "Sugarcane - Red Rot", "Sugarcane - Healthy", "Tomato - Bacterial Spot", "Tomato - Early Blight",
        "Tomato - Late Blight", "Tomato - Yellow Leaf Curl Virus", "Tomato - Healthy", "Wheat - Brown Rust",
        "Wheat - Yellow Rust", "Wheat - Healthy"
    ]
    num_classes = len(class_names)

    # Load model
    model_path = "D:\\Agriculture\\notebook\\kaggle1\\efficientnet_plant_disease_detection_model_state_dict (1).pth"
    model = load_model(model_path, num_classes)

    transform = get_transform()

    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

        # Make prediction
        prediction, confidence = predict_disease(model, image, class_names, transform)

        # Display result
        st.success(f"**Predicted Disease**: {prediction}")
        st.info(f"**Confidence Level**: {confidence:.2%}")


if __name__ == "__main__":
    main()

# to run this:
# streamlit run C:\Users\YOGESHWARAN\PycharmProjects\Plant\plant_disease_detection_app.py