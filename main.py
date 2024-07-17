import streamlit as st
from login import login_page
from scanner import main_page
from utils import add_custom_css

def main():
    add_custom_css()
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_page()
    else:
        main_page()

if __name__ == "__main__":
    main()
