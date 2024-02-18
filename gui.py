# Model Imports
import numpy as np
import tensorflow as tf
from tensorflow import keras

# GUI Imports
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# Model Loading
gender_mapping = ["Male", "Female"]
age_model = keras.models.load_model("./Weights/Age.h5")
gender_model = keras.models.load_model("./Weights/Gender.h5")

# Initialize GUI
top = Tk()
top.geometry('800x600')
top.title("Age and Gender Detection System")
top.configure(background='#CDCDCD')

# Defining Labels to be displayed
age_label = Label(top, background="#CDCDCD", font=('arial', 15, 'bold'))
gender_label = Label(top, background="#CDCDCD", font=('arial', 15, 'bold'))
image_input = Label(top)

def load_image(file_path):
    """
    Load and preprocess the image from the file path.

    Args:
    - file_path (str): The path to the input image file.

    Returns:
    - numpy.ndarray: The preprocessed image.
    """
    img = Image.open(file_path)
    img = img.resize((224, 224))
    img = np.expand_dims(img, axis=0)
    img = np.delete(np.array(img), 0, 1)
    img = np.resize(img, (224, 224, 3))
    img = np.reshape(img, (-1,224,224,3))
    img = img/255.
    return img

def detect_age_and_gender(file_path):
    """
    Perform age and gender detection on the given image file path.

    Args:
    - file_path (str): The path to the input image file.

    Returns:
    None
    """
    img = load_image(file_path)

    # Predict age and gender
    age = float(tf.round(tf.squeeze(age_model.predict(img, verbose=0)), 2))
    gender = gender_mapping[int(tf.round(tf.squeeze(gender_model.predict(img, verbose=0))))]

    # Display the results
    print("Age: {} Gender: {}".format(age, gender))
    age_label.configure(foreground="#011638", text="Age: {}".format(age))
    gender_label.configure(foreground="#011638", text="Gender: {}".format(gender))

def show_detect_btn(file_path):
    """
    Show the 'Detect Image' button after an image is uploaded.

    Args:
    - file_path (str): The path to the input image file.

    Returns:
    None
    """
    detect_btn = Button(top, 
        text="Detect Image", 
        command=lambda: detect_age_and_gender(file_path),
        padx=10, pady=5
    )

    detect_btn.configure(
        background="#364156",
        foreground="white",
        font=('arial', 10, 'bold')
    )

    detect_btn.place(relx=0.79, rely=0.46)

def upload_image():
    """
    Open a file dialog to upload an image and display it.

    Args:
    None

    Returns:
    None
    """
    try:
        filepath = filedialog.askopenfilename()
        uploaded = Image.open(filepath)
        uploaded.thumbnail((
            (top.winfo_width()/2.25), (top.winfo_height()/2.25)
        ))

        image = ImageTk.PhotoImage(uploaded)
        image_input.configure(image=image)
        image_input.image = image
        age_label.configure(text='')
        gender_label.configure(text='')
        show_detect_btn(filepath)

    except Exception as e:
        print("Error:", e)

# Detecting Age and Gender
upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground="white", font=('arial', 10, 'bold'))
upload.pack(side="bottom", expand=True)

# Visual Setting
image_input.pack(side="bottom", expand=True)
age_label.pack(side="bottom", expand=True)
gender_label.pack(side="bottom", expand=True) 

# Set Heading / Title
heading = Label(top, text="Age and Gender Detection System", pady=20, font=('arial', 20, 'bold'))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

# Start program
top.mainloop()

# DEBUG
print("Done")
