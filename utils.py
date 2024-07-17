import streamlit as st
import pickle
from pathlib import Path
from argon2 import PasswordHasher
from pyzbar import pyzbar

def add_custom_css():
    st.markdown("""
    <style>
    
    .big-font {
        font-size: 24px;
        font-weight: bold;
        color: #4A4A4A;
    }
    .animated-text {
        font-size: 20px;
        font-weight: bold;
        background: linear-gradient(45deg, #ff00ff, #00ffff, #ff00ff);
        background-size: 200% 200%;
        color: white;
        padding: 10px;
        border-radius: 5px;
        animation: gradient 5s ease infinite;
    }
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .stButton > button {
        color: #ffffff;
        background-color: #4CAF50;
        border: none;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
        border-radius: 12px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .import streamlit as st

    <style>

    .custom-radio {
        background-color: #f0f8ff;  /* Light blue background */
        padding: 10px;
        border-radius: 10px;
        border: 2px solid #4CAF50;  /* Green border */
        margin-bottom: 10px;
    }
    .custom-radio .st-bc {
        background-color: transparent;
    }
    .custom-radio .st-bx {
        color: #4CAF50;  /* Green text */
    }
    </style>
    """, unsafe_allow_html=True)


def save_hashed_passwords(passwords):
    ph = PasswordHasher()
    hashed_passwords = [ph.hash(p) for p in passwords]
    file_path = Path(__file__).parent / "hashed_pw.pkl"
    with file_path.open("wb") as file:
        pickle.dump(hashed_passwords, file)

def verify_password(username, password, usernames):
    ph = PasswordHasher()
    file_path = Path(__file__).parent / "hashed_pw.pkl"
    with file_path.open("rb") as file:
        stored_hashes = pickle.load(file)
    if username in usernames:
        idx = usernames.index(username)
        try:
            return ph.verify(stored_hashes[idx], password)
        except:
            return False
    return False

def scan_image_for_codes(image_np):
    decoded_objects = pyzbar.decode(image_np)
    return decoded_objects