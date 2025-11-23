# ============================================================
# Streamlit – FaceNet Face ID (Open-set with Unknown Rejection)
# ============================================================
import os, pickle, numpy as np
from PIL import Image
import streamlit as st
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

# Paths
ART_DIR = r"C:\Users\vishw\OneDrive\Desktop\Face_identification\facenet_artifacts"
CLF_PKL = os.path.join(ART_DIR, "clf.pkl")
LE_PKL  = os.path.join(ART_DIR, "label_encoder.pkl")
# Optional: if you ever fine-tune FaceNet, load weights you saved; otherwise kept None
EMBEDDER_PTH = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="FaceNet Face ID", page_icon="🧠", layout="centered")
st.title("🧠 Face Identification (FaceNet + Classifier)")
st.caption("Detects & aligns face → 512D embedding → predicts with unknown rejection.")

# Sidebar
THRESH = st.sidebar.slider("Unknown rejection threshold (probability)", 0.1, 1.0, 0.6, 0.05)
st.sidebar.write(f"Device: **{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}**")

@st.cache_resource
def load_models():
    mtcnn = MTCNN(image_size=160, margin=14, device=DEVICE, post_process=True)
    embedder = InceptionResnetV1(pretrained='vggface2', classify=False).to(DEVICE).eval()
    if EMBEDDER_PTH and os.path.exists(EMBEDDER_PTH):
        state = torch.load(EMBEDDER_PTH, map_location=DEVICE)
        embedder.load_state_dict(state, strict=False)

    with open(CLF_PKL, "rb") as f: clf = pickle.load(f)
    with open(LE_PKL, "rb") as f:  le  = pickle.load(f)
    return mtcnn, embedder, clf, le

mtcnn, embedder, clf, le = load_models()

def predict(img: Image.Image):
    face = mtcnn(img, return_prob=False)
    if face is None:
        # fallback simple resize
        img2 = img.resize((160,160))
        face = torch.from_numpy(np.array(img2)).permute(2,0,1).float()/255.0
    face = face.unsqueeze(0).to(DEVICE)  # [1,3,160,160]
    with torch.no_grad():
        emb = embedder(face).cpu().numpy()   # [1,512]

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(emb)[0]
    else:
        dec = clf.decision_function(emb)[0]
        e = np.exp(dec - dec.max()); probs = e / e.sum()

    idx = int(probs.argmax()); p = float(probs[idx]); name = str(le.inverse_transform([idx])[0])
    return name, p, probs

uploaded = st.file_uploader("📤 Upload a face image", type=["jpg","jpeg","png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    preview = image.copy(); preview.thumbnail((420,420))
    st.image(preview, caption="Uploaded (resized preview)")

    if st.button("🚀 Predict"):
        with st.spinner("Detecting face, embedding and classifying..."):
            name, p, probs = predict(image)

        if p < THRESH:
            st.warning("❌ Unknown / Not a recognized face")
        else:
            st.success(f"✅ {name} — {p*100:.2f}%")

        st.markdown("### 🔍 Class probabilities")
        st.dataframe({"Class": list(le.classes_), "Prob (%)": [round(x*100,2) for x in probs]}, use_container_width=True)
else:
    st.info("Upload a face image to start.")
