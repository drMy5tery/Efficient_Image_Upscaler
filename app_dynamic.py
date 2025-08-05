import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from keras.config import enable_unsafe_deserialization
from streamlit_image_comparison import image_comparison
import io
from datetime import datetime

# -----------------------
# Custom Layers
# -----------------------
class SubpixelUpscale(Layer):
    def __init__(self, scale, **kwargs):
        super(SubpixelUpscale, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale)

    def compute_output_shape(self, input_shape):
        batch, h, w, c = input_shape
        return (batch,
                None if h is None else h * self.scale,
                None if w is None else w * self.scale,
                c // (self.scale ** 2))

    def get_config(self):
        return {"scale": self.scale}

class CastToFloat32(Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

    def get_config(self):
        return {}

# -----------------------
# App Configuration
# -----------------------
enable_unsafe_deserialization()
st.set_page_config(page_title="🧠 Super-Resolution Demo", layout="wide")
st.title("📸 Deep Learning-based Image Super-Resolution")

# -----------------------
# Sidebar Options
# -----------------------
st.sidebar.title("🛠️ Options")
mode = st.sidebar.radio("Choose Mode", ["🧪 Test Model", "✨ Enhance Image"])
scale = st.sidebar.selectbox("Upscale Factor", [2, 4, 8], index=0)

model_paths = {
    2: "espcn_model_x2.keras",
    4: "espcn_model_x4.keras",
    8: "espcn_model_x8.keras"
}

# Load model
model = load_model(model_paths[scale], compile=False,
                   custom_objects={'SubpixelUpscale': SubpixelUpscale, 'CastToFloat32': CastToFloat32})

# -----------------------
# Upload Section
# -----------------------
uploaded_file = st.file_uploader("📤 Upload an Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image).astype(np.float32) / 255.0
    orig_h, orig_w = img_np.shape[:2]

    st.info(f"📐 Uploaded Image Size: {orig_w} x {orig_h}")

    if mode == "🧪 Test Model":
        # Downscale the image
        lr = cv2.resize(img_np, (orig_w // scale, orig_h // scale), interpolation=cv2.INTER_CUBIC)
        sr = model.predict(np.expand_dims(lr, axis=0))[0]
        sr = np.clip(sr, 0, 1)

        # Resize for comparison
        target_shape = (sr.shape[1], sr.shape[0])
        original_resized = cv2.resize(img_np, target_shape, interpolation=cv2.INTER_CUBIC)
        lr_upscaled = cv2.resize(lr, target_shape, interpolation=cv2.INTER_CUBIC)

        # Metrics
        psnr_lr = psnr(original_resized, lr_upscaled, data_range=1.0)
        psnr_sr = psnr(original_resized, sr, data_range=1.0)
        ssim_lr = ssim(original_resized, lr_upscaled, data_range=1.0, channel_axis=-1)
        ssim_sr = ssim(original_resized, sr, data_range=1.0, channel_axis=-1)

        # Comparison Slider
        st.subheader("Low-Res vs Super-Resolved")
        image_comparison(
            img1=(lr_upscaled * 255).astype(np.uint8),
            img2=(sr * 255).astype(np.uint8),
            label1=f"Low-Res Upscaled (x{scale})",
            label2="Super-Resolved",
            width=700,
        )

        # Metrics Table
        st.subheader("📊 Quality Metrics")
        st.markdown(f"""
        | Metric        | Low-Res (x{scale}) | Super-Resolved |
        |---------------|--------------------|----------------|
        | **PSNR (dB)** | {psnr_lr:.2f}      | **{psnr_sr:.2f}** |
        | **SSIM**      | {ssim_lr:.4f}      | **{ssim_sr:.4f}** |
        """)

        # Gallery
        st.subheader("🖼️ Image Gallery")
        col1, col2, col3 = st.columns(3)
        col1.image((img_np * 255).astype(np.uint8), caption="Original HR", use_container_width=True)
        col2.image((lr_upscaled * 255).astype(np.uint8), caption="Low-Res (Upscaled)", use_container_width=True)
        col3.image((sr * 255).astype(np.uint8), caption="Super-Resolved", use_container_width=True)

    elif mode == "✨ Enhance Image":
        # Direct enhancement
        sr = model.predict(np.expand_dims(img_np, axis=0))[0]
        sr = np.clip(sr, 0, 1)
        upscaled_h, upscaled_w = sr.shape[:2]
        st.success(f"📈 Enhanced Resolution: {upscaled_w} x {upscaled_h}")

        # PSNR and SSIM between original and upscaled
        resized_original = cv2.resize(img_np, (upscaled_w, upscaled_h), interpolation=cv2.INTER_CUBIC)
        psnr_sr = psnr(resized_original, sr, data_range=1.0)
        ssim_sr = ssim(resized_original, sr, data_range=1.0, channel_axis=-1)

        # Compare side-by-side
        st.subheader("🔍 Original vs Enhanced")
        image_comparison(
            img1=(resized_original * 255).astype(np.uint8),
            img2=(sr * 255).astype(np.uint8),
            label1="Bicubic Upscaled Original",
            label2="Enhanced",
            width=700,
        )

        # Metrics
        st.subheader("📊 Similarity to Bicubic Upscaling(Just for reference , not much meaningful as we don't know the Ground Truth)")
        st.markdown(f"""
        | Metric        | Enhanced |
        |---------------|----------|
        | **PSNR (dB)** | {psnr_sr:.2f} |
        | **SSIM**      | {ssim_sr:.4f} |
        """)

        st.subheader("🖼️ Image Gallery")
        col1, col2 = st.columns(2)
        col1.image((img_np * 255).astype(np.uint8), caption="Original", use_container_width=True)
        col2.image((sr * 255).astype(np.uint8), caption="Enhanced", use_container_width=True)

    # -----------------------
    # Download Section
    # -----------------------
    download_label = "📥 Download Super Resolved Test Image" if mode == "🧪 Test Model" else "📥 Download Enhanced Image"
    st.subheader(download_label)
    sr_image = Image.fromarray((sr * 255).astype(np.uint8))
    buf = io.BytesIO()
    sr_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button("💾 Download", byte_im, file_name=f"super_res_{timestamp}.png", mime="image/png")
