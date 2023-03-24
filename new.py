import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from tkinter import *
from tkinter import filedialog

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Define a function to classify an image
def classify_image(img_path):
    # Load the image using OpenCV
    img = cv2.imread(img_path)
    # Resize the image to 224x224 (the input size of VGG16)
    img = cv2.resize(img, (224, 224))
    # Preprocess the image for input to the VGG16 model
    img = preprocess_input(img)
    # Expand the image dimensions to match the input shape of VGG16
    img = np.expand_dims(img, axis=0)
    # Use the VGG16 model to predict the image class
    preds = model.predict(img)
    # Decode the predictions to get the class labels
    labels = decode_predictions(preds, top=1)[0]
    # Return the top predicted label
    return labels[0][1]

# Define a function to handle button clicks in the GUI
def process_image():
    # Ask the user to select an image file
    file_path = filedialog.askopenfilename()
    # Classify the image using the pre-trained VGG16 model
    class_label = classify_image(file_path)
    # Update the GUI with the predicted class label
    result_label.config(text="Class: " + class_label)

# Create the GUI window
window = Tk()
window.title("Lost Pet Detector")

# Add a button to trigger image processing
process_button = Button(window, text="Process Image", command=process_image)
process_button.pack()

# Add a label to display the results
result_label = Label(window, text="No image processed yet.")
result_label.pack()

# Start the GUI event loop
window.mainloop()