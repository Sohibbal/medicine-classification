import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from PIL import ImageOps
import numpy as np
import base64

# --- Konfigurasi dasar ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.set_page_config(page_title="Medicine Classifier", page_icon="💊", layout="wide")

# --- Sidebar kiri ---
st.sidebar.title("💊 Medicine Classification App")
st.sidebar.header("How to use")
st.sidebar.markdown("""
1. Unggah gambar resep obat (.jpg, .jpeg, .png) 📤  
2. Tunggu proses prediksi oleh model 🔍  
3. Lihat hasil klasifikasi dan tingkat keyakinannya 💯  
""")
st.sidebar.info("💡 Pastikan gambar cukup jelas dan *berbentuk persegi (1:1)* agar hasil lebih akurat.")

st.sidebar.markdown("---")
st.sidebar.header("👩‍💻 About")
st.sidebar.markdown("""
Dibuat oleh *Mona Ramadhani*  
Instagram: [@monaramadhaniii](https://www.instagram.com/monaramadhaniii)
""")
st.sidebar.caption("Developed using Streamlit + PyTorch")

# --- Halaman utama ---
st.title("💊 Medicine Classification App 💊")

st.write("Unggah gambar resep dokter Anda untuk mendapatkan prediksi obat yang tertera:")

# === Upload & Prediksi ===
uploaded_file = st.file_uploader("📤 Pilih gambar resep obat...", type=["jpg", "jpeg", "png"])

# --- Load Model ---
@st.cache_resource
def load_model():
    checkpoint = torch.load("best_resnet18.pt", map_location=DEVICE)
    class_names = checkpoint["label_map"]

    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(class_names))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    model.to(DEVICE)
    return model, class_names

model, CLASS_NAMES = load_model()

# --- Transformasi gambar ---
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def resize_with_padding(image, target_size=(320, 320)):
    """
    Resize gambar jadi ukuran target (320x320) tanpa mengubah rasio aslinya,
    dengan menambah padding sesuai warna background gambar asli.
    """
    # konversi ke RGB kalau belum
    if image.mode != "RGB":
        image = image.convert("RGB")

    # ambil warna rata-rata dari tepi gambar (asumsi background)
    np_img = np.array(image)
    edge_color = np.mean([
        np_img[0, :, :],          # atas
        np_img[-1, :, :],         # bawah
        np_img[:, 0, :],          # kiri
        np_img[:, -1, :]          # kanan
    ], axis=(0, 1)).astype(np.uint8)
    bg_color = tuple(edge_color)

    # resize gambar dengan mempertahankan rasi
    image.thumbnail(target_size, Image.Resampling.LANCZOS)

    # tambahkan padding agar jadi ukuran target_size
    delta_w = target_size[0] - image.size[0]
    delta_h = target_size[1] - image.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    new_im = ImageOps.expand(image, padding, fill=bg_color)
    return new_im

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Gambar diunggah", width=400)

    if image.mode != "RGB":
        image = image.convert("RGB")

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with st.spinner("🔍 Sedang memprediksi..."):
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = CLASS_NAMES[pred_idx]
        confidence = probs[0, pred_idx].item() * 100

    st.success(f"*Prediksi:* {pred_class}")
    st.caption(f"Tingkat keyakinan: {confidence:.2f}%")

    top5_prob, top5_idx = torch.topk(probs, 5)
    top5_text = "\n".join(
    [f"{i+1}. {CLASS_NAMES[top5_idx[0][i]]} — {top5_prob[0][i]*100:.2f}%" for i in range(5)]
)
    with st.expander("🔝 Top 5 Prediksi: "):
        st.write(top5_text)

# --- Info Model ---
with st.expander("📈 Informasi Model"):
    st.write("""
    Model ini menggunakan arsitektur *ResNet-18 pretrained ImageNet* dengan optimisasi *Adam* dan fungsi loss *CrossEntropyLoss*.         
    Dilatih untuk mengenali tulisan tangan pada resep dokter dan memprediksi nama obat yang tertera.

    *Akurasi model: 95%*     
    Model ini telah diuji menggunakan dataset resep dokter dan menunjukkan performa yang sangat baik dalam mengenali nama obat.

             
    *Parameter utama:*
    - Learning Rate: 0.001  
    - Weight Decay: 1e-4  
    - Epochs: 20  
    - Early Stopping: 5 epoch  
             
    """)
    st.caption("""
    🧠 *Model ini dilatih untuk mengenali 75 jenis nama obat*, yaitu:  
    Beklo, Maxima, Leptic, Esoral, Omastin, Esonix, Canazole, Fixal, Progut, Diflu, Montair, Flexilax, Maxpro, Vifas, Conaz, Fexofast, Fenadin, Telfast, Dinafex, Ritch, Renova, Flugal, Axodin, Sergel, Nexum, Opton, Nexcap, Fexo, Montex, Exium, Lumona, Napa, Azithrocin, Atrizin, Monas, Nidazyl, Metsina, Baclon, Rozith, Bicozin, Ace, Amodis, Alatrol, Napa Extend, Rivotril, Montene, Filmet, Aceta, Tamen, Bacmax, Disopan, Rhinil, Flamyd, Metro, Zithrin, Candinil, Lucan-R, Backtone, Bacaid, Etizin, Az, Romycin, Azyth, Cetisoft, Dancel, Tridosil, Nizoder, Ketoral, Ketocon, Ketotab, Ketozol, Denixil, Provair, Odmon, Baclofen, MKast, dan Trilock, Flexibac.
    """)
    st.info("📂 File model tersimpan di best_resnet18.pt")