import streamlit as st
import pickle
from pathlib import Path
from argon2 import PasswordHasher
from pyzbar import pyzbar

def add_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }
        
        @keyframes gradientBG {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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
