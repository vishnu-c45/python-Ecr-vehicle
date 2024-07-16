import pytesseract
from PIL import Image
import os

# Set the Tesseract executable path if necessary (for Windows users)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Define the image path
image_path = 'test2.jpg'

# Debugging step: Check if the file exists and print its absolute path
if not os.path.isfile(image_path):
    raise FileNotFoundError(f"The file {image_path} does not exist.")
else:
    print(f"The file {image_path} exists.")
    print(f"Absolute path: {os.path.abspath(image_path)}")

# Debugging step: Check file permissions
if not os.access(image_path, os.R_OK):
    raise PermissionError(f"The file {image_path} is not readable.")

# Open the image file
try:
    img = Image.open(image_path)
    print("img", img)
except Exception as e:
    raise PermissionError(f"An error occurred while opening the image: {e}")

# Extract text from the image
try:
    text = pytesseract.image_to_string(img, lang='eng')
    print("text", text)
except Exception as e:
    raise RuntimeError(f"An error occurred while extracting text from the image: {e}")
