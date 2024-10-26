import streamlit as st
from pathlib import Path
from utils import add_custom_css, save_hashed_passwords, verify_password

def login_page():
    add_custom_css()
    st.markdown('<div class="login-form">', unsafe_allow_html=True)
    st.title("Login Form")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    names = ["Yogeshvar", "Leann", "Augustine"]
    usernames = ["Yogeshvar", "Leann", "Augustine"]
    passwords = ["a123", "b123", "c123"]

    if not Path(__file__).parent.joinpath("hashed_pw.pkl").exists():
        save_hashed_passwords(passwords)

    if st.button("Login"):
        if username and password:
            if verify_password(username, password, usernames):
                st.success(f"Welcome {username}!")
                st.session_state.logged_in = True
            else:
                st.error("Incorrect username or password. Please try again.")
        else:
            st.error("Please enter both username and password")
    st.markdown('</div>', unsafe_allow_html=True)
