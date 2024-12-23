from pdf2image import convert_from_path
import pytesseract
import os

from config import INPUT_DIR, OUTPUT_DIR

# Function to convert PDF to text using Tesseract OCR
def pdf_to_text(pdf_path, output_txt_path, dpi=300, lang='eng'):
    """
    Convert a PDF file to text using Tesseract OCR and save the output to a text file.

    Parameters:
        pdf_path (str): Path to the input PDF file.
        output_txt_path (str): Path to the output text file.
        dpi (int): DPI for image conversion. Default is 300.
        lang (str): Language for OCR. Default is English ('eng').
    """
    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path, dpi=dpi)

        # Initialize an empty string to store the extracted text
        extracted_text = ""

        # Process each image using Tesseract
        for i, image in enumerate(images):
            print(f"Processing page {i + 1}...")
            text = pytesseract.image_to_string(image, lang=lang)
            extracted_text += text + '\n'  # Add a newline between pages

        # Write the extracted text to the output file
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(extracted_text)

        print(f"Text successfully saved to {output_txt_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    
    for file in os.listdir(INPUT_DIR):
        
    
        # Replace with your file paths
        pdf_path = INPUT_DIR + file  # Path to the input PDF file
        output_txt_path = OUTPUT_DIR  # Path to the output text file

        # Call the function
        name = file.split('.')[0]
        pdf_to_text(pdf_path, output_txt_path + name + '.txt')