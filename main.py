import streamlit as st
from login import login_page
from scanner import main_page
from training import train_and_evaluate, plot_metrics  # Importing functions from training.py

# Streamlit app configuration
if __name__ == "__main__":
    st.set_page_config(page_title="Barcode/QR & Training App")

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'next_page' not in st.session_state:
        st.session_state.next_page = False

    # Handle login
    if not st.session_state.logged_in:
        login_page()
    else:
        # Sidebar options for navigation
        page = st.sidebar.selectbox("Navigate", ["Scanner", "Training"])

        if page == "Scanner":
            main_page()  # Calls the scanner page
        elif page == "Training":
            # Include the training page for CNN model training
            st.title("QR Code/Barcode Classification with CNN")
            
            # Upload training and testing CSV files
            train_csv_file = st.file_uploader("Upload Training Dataset CSV", type=["csv"])
            test_csv_file = st.file_uploader("Upload Testing Dataset CSV", type=["csv"])

            if train_csv_file and test_csv_file:
                if st.button("Train and Evaluate Model"):
                    history, accuracy, y_test, y_pred, y_pred_prob = train_and_evaluate(train_csv_file, test_csv_file)
                    if accuracy:
                        st.success(f"Model trained successfully! Test Accuracy: {accuracy:.2f}%")
                        plot_metrics(history, y_test, y_pred, y_pred_prob)  # Display training graphs
            else:
                st.write("Please upload both training and testing CSV files to start.")
