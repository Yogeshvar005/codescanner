import io
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode
import webbrowser
from datetime import datetime
import time
import qrcode
from barcode import Code128
from barcode.writer import ImageWriter
from io import BytesIO

def new_page():
    st.title("QR Code and Barcode Generator")
    option = st.selectbox("Select Code Type", ["QR Code", "Barcode"])
    if option == "QR Code":
        text = st.text_input("Enter text for QR Code")
        if st.button("Generate QR Code"):
            if text:
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_L,
                    box_size=10,
                    border=4,
                )
                qr.add_data(text)
                qr.make(fit=True)
                img = qr.make_image(fill_color="black", back_color="white")
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                byte_im = buf.getvalue()
                st.image(byte_im, caption="Generated QR Code")
                st.download_button(
                    label="Download QR Code",
                    data=byte_im,
                    file_name="qrcode.png",
                    mime="image/png"
                )
            else:
                st.warning("Please enter text to generate QR Code")

    elif option == "Barcode":
        text = st.text_input("Enter text for Barcode")
        if st.button("Generate Barcode"):
            if text:
                barcode = Code128(text, writer=ImageWriter())
                buf = BytesIO()
                barcode.write(buf)
                byte_im = buf.getvalue()
                st.image(byte_im, caption="Generated Barcode")
                st.download_button(
                    label="Download Barcode",
                    data=byte_im,
                    file_name="barcode.png",
                    mime="image/png"
                )
            else:
                st.warning("Please enter text to generate Barcode")

def beep():
    # Replace winsound with a Streamlit alternative
    st.audio(data=b'\x00\x00' + b'\xff\x7f' * 44100, sample_rate=44100)

def main_page():
    st.sidebar.title("Navigation")
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4481/4481064.png", use_column_width=True)
    option = st.sidebar.selectbox("Select an option", ["Home", "Upload Image", "Barcode", "QR Code", "Generator"])

    if option == "Home":
        st.title("Welcome to CodeScanner")
        st.markdown('<p class="big-font">Scan barcodes and QR codes with ease!</p>', unsafe_allow_html=True)

    elif option == "Upload Image":
        st.title("Upload Image")
        st.markdown('<p class="animated-text">Upload an image to scan for barcodes or QR codes.</p>', unsafe_allow_html=True)
        st.markdown('<div class="custom-radio">', unsafe_allow_html=True)
        open_in = st.radio("Open detected code in:", ["Google", "Amazon"])
        st.markdown('</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            with st.spinner("Scanning for codes..."):
                time.sleep(2)  # Simulating processing time
                decoded_objects = scan_image_for_codes(image_np)
            
            if decoded_objects:
                st.success("Codes detected!")
                for obj in decoded_objects:
                    st.info(f"Detected {obj.type}: {obj.data.decode('utf-8')}")
                    data = obj.data.decode('utf-8')
                    if st.button(f"Open in {open_in}", key=data):
                        if open_in == "Google":
                            st.markdown(f"[Open in Google](https://www.google.com/search?q={data})")
                        elif open_in == "Amazon":
                            st.markdown(f"[Open in Amazon](https://www.amazon.com/s?k={data})")
            else:
                st.warning("No codes detected in the image.")

    elif option == "Barcode":
        st.title("Barcode Scanner")
        st.markdown('<p class="animated-text">Scan barcodes in real-time!</p>', unsafe_allow_html=True)
        st.markdown('<div class="custom-radio">', unsafe_allow_html=True)
        open_in = st.radio("Open detected code in:", ["Google", "Amazon"])
        st.markdown('</div>', unsafe_allow_html=True)       
        display_datetime = st.checkbox("Display detection date and time")
        if st.button("Start Barcode Scanner", key='start_barcode'):
            st.session_state.camera_active = True
            run_camera("barcode", open_in, display_datetime)

    elif option == "QR Code":
        st.title("QR Code Scanner")
        st.markdown('<p class="animated-text">Scan QR codes instantly!</p>', unsafe_allow_html=True)
        st.markdown('<div class="custom-radio">', unsafe_allow_html=True)
        open_in = st.radio("Open detected code in:", ["Google", "Amazon"])
        st.markdown('</div>', unsafe_allow_html=True)        
        display_datetime = st.checkbox("Display detection date and time")
        if st.button("Start QR Code Scanner", key='start_qr'):
            st.session_state.camera_active = True
            run_camera("qrcode", open_in, display_datetime)

    elif option == "Generator":
        new_page()

def run_camera(code_type, open_in, display_datetime):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    stop_button = st.button("Stop Scanner")
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    while st.session_state.camera_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break
        
        decoded_objects = decode(frame)
        for obj in decoded_objects:
            detected_code = obj.data.decode('utf-8')
            st.success(f"{code_type.capitalize()} detected: {detected_code}")
            if display_datetime:
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.info(f"Detected at: {current_datetime}")
            beep()
            if open_in == "Google":
                st.markdown(f"[Open in Google](https://www.google.com/search?q={detected_code})")
            elif open_in == "Amazon":
                st.markdown(f"[Open in Amazon](https://www.amazon.com/s?k={detected_code})")
            st.session_state.camera_active = False
            break
        
        # Add a scanning effect
        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text(f"Scanning... {i+1}%")
            time.sleep(0.01)
        
        stframe.image(frame, channels="BGR", use_column_width=True)
        if stop_button:
            st.session_state.camera_active = False
            break
    
    cap.release()
    cv2.destroyAllWindows()
    stframe.empty()
    progress_bar.empty()
    status_text.empty()

def scan_image_for_codes(image):
    decoded_objects = decode(image)
    return decoded_objects
