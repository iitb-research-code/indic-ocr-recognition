import streamlit as st
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import os

# Set the path for Tesseract (adjust if not in the system's PATH)
# For Windows users, you may need to specify the path to the tesseract executable like:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to convert PDF to images
def pdf_to_images(pdf_file):
    # Convert the PDF file to images (one per page)
    images = convert_from_path(pdf_file)
    return images


tesseract_langs = {
    "assamese": "asm",
    "bengali": "ben",
    "gujarati": "guj",
    "punjabi": "pan",
    "hindi": "hin",
    "sanskrit": "san",
    "sindhi": "snd",
    "kannada": "kan",
    "malayalam": "mal",
    "manipuri": "mni",
    "marathi": "mar",
    "odia": "ori",
    "tamil": "tam",
    "telugu": "tel",
    "urdu": "urd",
    "nepali": "nep",
    "english": "eng",
    "math": "equ"

}

# Streamlit app
def main():
    st.title("Tesseract OCR with PDFs and Language Selection")

    st.write(
        """
        Upload a PDF and choose a language for Tesseract OCR to extract text from the pages of the PDF.
        """
    )

    languages = ['english','assamese', 'bengali', 'gujarati', 'punjabi', 'hindi', 'kannada', 'malayalam', 'manipuri', 'marathi', 'odia', 'tamil', 'telugu', 'urdu', 'equ']

    


    selected_lang = st.selectbox("Choose language for OCR", languages)

    # PDF file upload
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Save the uploaded PDF to a temporary location
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Convert the PDF to images
        images = pdf_to_images("temp.pdf")
        
        # Display the images for each page
        for i, image in enumerate(images):
            st.image(image, caption=f'Page {i+1}', use_column_width=True)
            
            # Run Tesseract OCR to extract text from the image with the selected language
            text = pytesseract.image_to_string(image, lang=tesseract_langs[selected_lang])
            
            st.subheader(f"Extracted Text from Page {i+1}:")
            if text:
                st.write(text)
            else:
                st.write("No text found on this page.")

        # Clean up the temporary file
        os.remove("temp.pdf")

if __name__ == "__main__":
    main()
