import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pyzbar import pyzbar
import webbrowser
import winsound
from datetime import datetime
import base64

def new_page():
    st.title("New Page")
    st.write("Welcome to the new page!")
    st.write("This is a new page.")

def beep():
    winsound.Beep(2500, 500)

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode("utf-8")
    return b64_string

def main_page():
    st.sidebar.title("")
    
    # Encode the image as a base64 string and display it in the sidebar
    sidebar_image_base64 = get_base64_image("sidebar.jpg")
    st.sidebar.markdown(
        f"""
        <img src="data:image/jpg;base64,{sidebar_image_base64}" alt="sidebar image" width="100%">
        """,
        unsafe_allow_html=True
    )
    
    option = st.sidebar.selectbox("Select an option", [" ", "Upload Image", "Barcode", "QR Code"])

    st.markdown(
        """
        <style>
        .stApp {
            background: url("https://img.freepik.com/free-photo/artistic-blurry-colorful-wallpaper-background_58702-8628.jpg?size=626&ext=jpg") no-repeat center fixed;
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if option == "Upload Image":
        st.title("Upload Image")
        st.write("Upload an image to scan for barcodes or QR codes.")
        open_in = st.radio("Open detected code in:", ["Google", "Amazon"])
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Scanning for barcodes and QR codes...")
            decoded_objects = scan_image_for_codes(image_np)
            for obj in decoded_objects:
                st.write(f"Detected {obj.type}: {obj.data.decode('utf-8')}")
                data = obj.data.decode('utf-8')
                if open_in == "Google":
                    webbrowser.open(f"https://www.google.com/search?q={data}")
                elif open_in == "Amazon":
                    webbrowser.open(f"https://www.amazon.com/s?k={data}")

    elif option == "Barcode":
        st.title("Barcode Page")
        st.write("Welcome to the barcode page!")
        open_in = st.radio("Open detected code in:", ["Google", "Amazon"])
        display_datetime = st.checkbox("Display detection date and time")
        st.write("Open the camera to scan barcodes.")
        if st.button("Start Barcode Scanner"):
            if 'camera_active' not in st.session_state:
                st.session_state.camera_active = False
            if not st.session_state.camera_active:
                st.session_state.camera_active = True
                run_camera("barcode", open_in, display_datetime)

    elif option == "QR Code":
        st.title("QR Code Page")
        open_in = st.radio("Open detected code in:", ["Google", "Amazon"])
        display_datetime = st.checkbox("Display detection date and time")
        st.write("Welcome to the QR code page!")
        st.write("Open the camera to scan QR codes.")
        if st.button("Start QR Code Scanner"):
            if 'camera_active' not in st.session_state:
                st.session_state.camera_active = False
            if not st.session_state.camera_active:
                st.session_state.camera_active = True
                run_camera("qrcode", open_in, display_datetime)



def run_camera(code_type, open_in, display_datetime):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    stop_button = st.button("Stop Scanner")
   
    while st.session_state.camera_active:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture image")
            break
        decoded_objects = pyzbar.decode(frame)
        for obj in decoded_objects:
            detected_code = obj.data.decode('utf-8')
            st.write(f"{code_type.capitalize()} detected: {detected_code}")
            if display_datetime:
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.write(f"Detected at: {current_datetime}")
            beep()
            if open_in == "Google":
                webbrowser.open(f"https://www.google.com/search?q={detected_code}")
            elif open_in == "Amazon":
                webbrowser.open(f"https://www.amazon.com/s?k={detected_code}")
            st.session_state.camera_active = False
            break
        stframe.image(frame, channels="BGR")
        if stop_button:
            st.session_state.camera_active = False
            break
    cap.release()
    cv2.destroyAllWindows()
    stframe.empty()

def scan_image_for_codes(image):
    decoded_objects = pyzbar.decode(image)
    return decoded_objects

if __name__ == "_main_":
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    main_page()
