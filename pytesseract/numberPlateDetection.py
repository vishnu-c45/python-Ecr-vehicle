import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import os
import glob
import re
import mysql.connector
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
# Read the image
def numberPLateDetection():
    img = cv2.imread('image4.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Display the gray image
    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    plt.title('Gray Image')
    plt.show()

    # Noise reduction and edge detection
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    # Display the edges
    plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    plt.title('Edge Detection')
    plt.show()

    # Find contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is not None:
        # Create mask and extract the number plate region
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        # Display the masked image
        plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        plt.title('Masked Image with Number Plate')
        plt.show()

        # Crop the number plate from the gray image
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

        # Display the cropped image
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.title('Cropped Number Plate')
        # plt.show()

        # OCR to read the text from the number plate
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        
        if result:
            text = result[0][-2]
            print("Detected Number Plate Text:", text)
            font = cv2.FONT_HERSHEY_SIMPLEX
            res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)

            # Display the final image with the detected number plate
            plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
            plt.title('Final Image with Detected Number Plate')
            plt.show()
        else:
            print("No text detected by OCR.")
    else:
        print("No suitable contour found for the number plate.")
        



class ImageHandler(FileSystemEventHandler):
    print("In class function")
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = event.src_path
        if file_path.endswith('.jpg'):
            # image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
            print("file path",file_path)
            detect_number_plate(file_path)
            

def detect_number_plate(image_path):
    print("in detect function")
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    plt.show()

    # Noise reduction and edge detection
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    plt.show()
    

    # Find contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            print("length =4")
            location = approx
            break

    if location is not None:
        print("location is not None")
        # Create mask and extract the number plate region
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)
        plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        plt.show()
        

        # Crop the number plate from the gray image
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.show()

        # OCR to read the text from the number plate
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        print("@@@ end location")

        if result:
            print("result")
            # text = result[0][-2]
            # print(f"Detected Number Plate Text for {image_path}: {text}")  # Print the detected text
            raw_text = result[0][-2]
            text = re.sub(r'\W+', '', raw_text)  # Remove all non-alphanumeric characters
            print(f"Detected Number Plate Text for {image_path}: {text}")  # Print the cleaned text
            if insert_into_db(text):
               os.remove(image_path)
               print(f"Deleted {image_path} after successful DB insert.")
            else:
                print(f"Failed to insert {text} into the database.")

            font = cv2.FONT_HERSHEY_SIMPLEX
            res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)

            # Display the final image with the detected number plate
            plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
            plt.title(f'Final Image with Detected Number Plate for {image_path}')
            plt.show()
        else:
            print(f"No text detected by OCR for {image_path}.")
    else:
        print(f"No suitable contour found for the number plate in {image_path}.")




def insert_into_db(vehicle_number):
    # Database connection details
    print("in db")
    config = {
        'user': 'tis',        
        'password': 'tis',       
        'host': 'localhost',      
        'database': 'vehicleAi',  
        'raise_on_warnings': True
    }
    try:
        # Connect to the database
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Insert the vehicle number
        cursor.execute('INSERT INTO vehicle_numbers (vehicle_number,created_on) VALUES (%s,%s)', (vehicle_number,current_datetime))
        conn.commit()
        conn.close()
        print(f"Inserted {vehicle_number} into the database.")
        return True
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return False
    
folder_path = '/home/tis/pythonProjects/pytesseract/images'  

# Fetch all image files in the folder
image_files = glob.glob(os.path.join(folder_path, '*.jpg'))

# Process each image in the list
for image_file in image_files:
    print("image filers",image_file)
    detect_number_plate(image_file)




# Set up the watchdog observer
# event_handler = ImageHandler(folder_path)
# observer = Observer()
# observer.schedule(event_handler, folder_path, recursive=False)
# observer.start()

# try:
#     print(f"Monitoring folder: {folder_path}")
#     while True:
#         # Keep the script running
#         pass
# except KeyboardInterrupt:
#     observer.stop()
# observer.join()