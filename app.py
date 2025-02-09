import streamlit as st
from PIL import Image
import torch
from transformers import GPT2Tokenizer
from src.model import VisionLanguageCaptioningModel
from src.utils import load_model, load_image

# Streamlit Title
st.title("AI-Based Meme Caption Generator")

# Load the trained model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = VisionLanguageCaptioningModel()
model_path = "models/meme_captioning_model.pth"
model = load_model(model, model_path)
model.to(device)
model.eval()

# Image Upload UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate Caption Button
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            # Preprocess image and generate caption
            image_tensor = load_image(image).to(device).unsqueeze(0)
            
            input_ids = tokenizer.encode("Caption this meme: ", return_tensors="pt").to(device)
            attention_mask = torch.ones_like(input_ids).to(device)

            # Forward pass to generate caption
            with torch.no_grad():
                outputs = model(image_tensor, input_ids, attention_mask)
                predicted_tokens = torch.argmax(outputs, dim=-1)
                caption = tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)

        st.success("Generated Caption:")
        st.write(caption)
