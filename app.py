import io
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =========================
# CONFIG A ADAPTER
# =========================
MODEL_PATH = Path("models/bjj_model.pth")  # <-- change si ton fichier a un autre nom
IMAGE_SIZE = 224

# IMPORTANT: mets EXACTEMENT la même normalisation que dans train.py
# (si tu as utilisé ImageNet/MobileNet standard, c'est bien ça)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Si tu veux forcer l'ordre des classes, mets-le ici.
# Sinon, on lit dataset/ pour reconstruire l'ordre (tri alphabétique).
DATASET_DIR = Path("dataset")


def get_classes():
    if DATASET_DIR.exists():
        classes = sorted([p.name for p in DATASET_DIR.iterdir() if p.is_dir()])
        if classes:
            return classes
    # fallback si dataset pas présent (déploiement)
    return ["back_control", "close_guard", "mount", "side_control"]


def build_model(num_classes: int):
    # MobileNetV2 (rapide, standard)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def load_model(model_path: Path, classes):
    device = torch.device("cpu")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    num_classes = len(classes)
    model = build_model(num_classes).to(device)
    model.eval()

    obj = torch.load(model_path, map_location=device)

    # Cas 1: tu as torch.save(model.state_dict(), path)
    if isinstance(obj, dict) and any(k.startswith("features") or k.startswith("classifier") for k in obj.keys()):
        model.load_state_dict(obj)

    # Cas 2: tu as torch.save({"state_dict": ..., "classes": ...}, path)
    elif isinstance(obj, dict) and "state_dict" in obj:
        model.load_state_dict(obj["state_dict"])

    # Cas 3: tu as torch.save(model, path) (full model)
    else:
        model = obj
        model.eval()

    return model


def get_preprocess():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@torch.inference_mode()
def predict_pil(model, img: Image.Image, classes):
    preprocess = get_preprocess()
    x = preprocess(img.convert("RGB")).unsqueeze(0)  # [1,3,H,W]
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)  # [C]
    topk = min(3, len(classes))
    vals, idxs = torch.topk(probs, k=topk)
    results = [(classes[i], float(vals[j])) for j, i in enumerate(idxs.tolist())]
    return results, probs.tolist()


# =========================
# UI
# =========================
st.set_page_config(page_title="BJJ Position Classifier", page_icon="🥋", layout="centered")

st.title("🥋 BJJ Stats — Position Classifier")
st.write("Upload une image, je prédis la position (MobileNet).")

classes = get_classes()

# Charge le modèle une fois (cache Streamlit)
@st.cache_resource
def cached_model():
    return load_model(MODEL_PATH, classes)

try:
    model = cached_model()
except Exception as e:
    st.error(f"Impossible de charger le modèle : {e}")
    st.stop()

uploaded = st.file_uploader("Image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(io.BytesIO(uploaded.read()))
    # Streamlit te prévient que use_container_width devient width -> on met width="stretch"
    st.image(img, caption="Image uploadée", width="stretch")

    results, probs = predict_pil(model, img, classes)

    st.subheader("Résultat")

    # Affiche top 3
    for label, p in results:
        st.write(f"**{label}** — {p:.3f}")

    # Message intelligent basé sur la confiance du top 1
    top_label, top_prob = results[0]
    if top_prob > 0.9:
        st.success("Haute confiance du modèle.")
    elif top_prob > 0.6:
        st.info("Confiance modérée.")
    else:
        st.warning("Confiance faible : la position peut être ambiguë.")

    with st.expander("Voir toutes les probabilités"):
        for cls, p in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True):
            st.write(f"{cls}: {p:.4f}")

else:
    st.info("Upload une image pour lancer la prédiction.")