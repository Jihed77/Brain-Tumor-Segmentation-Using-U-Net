import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Segmentation de tumeurs cérébrales", layout="wide")

# Titre et description
st.title("🧠 Segmentation de tumeurs cérébrales par U‑Net")
st.markdown("Téléchargez une image IRM (format PNG, JPG, TIF) et le modèle générera automatiquement le masque de la tumeur.")

IMG_WIDTH = 256
IMG_HEIGHT = 256


def dice_coefficients(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

def iou(y_true, y_pred, smooth=1e-6):
    intersection = tf.keras.backend.sum(y_true * y_pred)
    total = tf.keras.backend.sum(y_true + y_pred)
    union = total - intersection
    return (intersection + smooth) / (union + smooth)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        'unet.keras',
        custom_objects={'dice_coefficients': dice_coefficients, 'iou': iou}
    )

model = load_model()
st.success("✅ Modèle chargé avec succès.")


def preprocess_image(uploaded_file):
    # Lire l'image avec OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img.shape[:2]   # (H, W) pour affichage ultérieur
    # Redimensionner et normaliser
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_normalized = img_resized.astype(np.float32) / 255.0
    # Ajouter la dimension batch
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch, original_shape, img_resized

uploaded_file = st.file_uploader("📤 Choisissez une image IRM", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file is not None:
    with st.spinner("🔍 Prétraitement de l'image..."):
        img_batch, original_shape, img_resized = preprocess_image(uploaded_file)
    
    with st.spinner("🧠 Segmentation en cours..."):
        pred_mask = model.predict(img_batch)[0, :, :, 0]   # sortie (256,256) probabilités
        
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📷 Image originale")
        st.image(img_resized, use_container_width=True)
    
    with col2:
        st.subheader("🎭 Carte de probabilité")
        fig, ax = plt.subplots()
        im = ax.imshow(pred_mask, cmap='hot', interpolation='nearest')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
    
    with col3:
        st.subheader("🧩 Masque binaire (seuil 0.5)")
        st.image(binary_mask * 255, use_container_width=True, clamp=True)
    
    st.download_button(
        label="💾 Télécharger le masque binaire (PNG)",
        data=cv2.imencode('.png', (binary_mask * 255).astype(np.uint8))[1].tobytes(),
        file_name="segmentation_mask.png",
        mime="image/png"
    )
else:
    st.info("👈 Veuillez télécharger une image IRM pour commencer.")