import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# Constants
IMG_WIDTH = 64
IMG_HEIGHT = 64
CHANNELS = 1  # Grayscale image
BATCH_SIZE = 32
EPOCHS = 10
BASE_IMAGE_PATH = r'D:/codescannerog/train'  # Default image path (modify based on your setup)

# Streamlit UI for file upload and dataset selection
st.title("QR Code/Barcode Classification with CNN")

# Upload training and testing CSV files
train_csv_file = st.file_uploader("Upload Training Dataset CSV", type=["csv"])
test_csv_file = st.file_uploader("Upload Testing Dataset CSV", type=["csv"])

def load_dataset(uploaded_file):
    """Loads the dataset from the uploaded CSV file."""
    try:
        data = pd.read_csv(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def preprocess_image(image_path):
    """Preprocess the image for CNN input."""
    try:
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))  # Resize image
        image = img_to_array(image)
        image = image / 255.0  # Normalize pixel values to [0,1]
        return image
    except Exception as e:
        st.error(f"Error processing image {image_path}: {str(e)}")
        return None

def prepare_data(data):
    """Prepares image data and labels for training/testing."""
    images = []
    labels = []

    for _, row in data.iterrows():
        image_path = row['filename']  # Assuming the column with image paths is named 'filename'
        label = 1 if row[' qr_code'] == 1 else 0  # Assuming 1 for QR code, 0 for Barcode
        
        image = preprocess_image(image_path)

        if image is not None:
            images.append(image)
            labels.append(label)

    # Convert images and labels into numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    return X, y

def build_cnn_model():
    """Builds a simple CNN model."""
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_and_evaluate(train_csv, test_csv):
    """Train the model and evaluate it on the test set."""
    # Load datasets
    train_data = load_dataset(train_csv)
    test_data = load_dataset(test_csv)

    if train_data is None or test_data is None:
        return None

    # Prepare the data (images and labels)
    st.write("Preprocessing data...")
    X_train, y_train = prepare_data(train_data)
    X_test, y_test = prepare_data(test_data)

    # Reshape the data to fit the CNN input (add the channel dimension)
    X_train = X_train.reshape(X_train.shape[0], IMG_WIDTH, IMG_HEIGHT, CHANNELS)
    X_test = X_test.reshape(X_test.shape[0], IMG_WIDTH, IMG_HEIGHT, CHANNELS)

    # Ensure labels are binary (in case they are not already)
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    # Build the CNN model
    model = build_cnn_model()

    st.write("Training the model, please wait...")
    progress_bar = st.progress(0)

    # Store history over all epochs
    all_history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

    # Train the model and store training history
    for epoch in range(EPOCHS):
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=1, verbose=0)
        progress_bar.progress((epoch + 1) / EPOCHS)

        # Accumulate history
        all_history['accuracy'].extend(history.history['accuracy'])
        all_history['val_accuracy'].extend(history.history['val_accuracy'])
        all_history['loss'].extend(history.history['loss'])
        all_history['val_loss'].extend(history.history['val_loss'])

    # Evaluate the model
    st.write("Evaluating the model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    # Make predictions on the test set
    y_pred_prob = model.predict(X_test).flatten()  # Get predicted probabilities
    y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

    return all_history, test_accuracy * 100, y_test, y_pred, y_pred_prob


def plot_metrics(history, y_test, y_pred, y_pred_prob):
    """Plots training & validation accuracy, loss, confusion matrix, ROC curve, and precision-recall curve."""
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs_range = range(len(acc))  # Adjust to the length of the history

    # Create a figure with subplots
    fig, axs = plt.subplots(5, 1, figsize=(10, 25))

    # Plot accuracy
    axs[0].plot(epochs_range, acc, label='Training Accuracy')
    axs[0].plot(epochs_range, val_acc, label='Validation Accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='lower right')
    axs[0].set_title('Training and Validation Accuracy')

    # Plot loss
    axs[1].plot(epochs_range, loss, label='Training Loss')
    axs[1].plot(epochs_range, val_loss, label='Validation Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc='upper right')
    axs[1].set_title('Training and Validation Loss')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=axs[2])
    axs[2].set_title('Confusion Matrix')

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    axs[3].plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
    axs[3].plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
    axs[3].set_xlabel('False Positive Rate')
    axs[3].set_ylabel('True Positive Rate')
    axs[3].set_title('Receiver Operating Characteristic (ROC) Curve')
    axs[3].legend(loc='lower right')

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    axs[4].plot(recall, precision, color='green')
    axs[4].set_xlabel('Recall')
    axs[4].set_ylabel('Precision')
    axs[4].set_title('Precision-Recall Curve')

    # Display plots in Streamlit
    st.pyplot(fig)


# Streamlit interface for training and evaluation
if train_csv_file and test_csv_file:
    if st.button("Train and Evaluate Model"):
        history, accuracy, y_test, y_pred, y_pred_prob = train_and_evaluate(train_csv_file, test_csv_file)
        if accuracy:
            st.success(f"Model trained successfully! Test Accuracy: {accuracy:.2f}%")
            plot_metrics(history, y_test, y_pred, y_pred_prob)  # Display training graphs
else:
    st.write("Please upload both training and testing CSV files to start.")
