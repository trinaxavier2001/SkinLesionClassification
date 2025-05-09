import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

# ─── CONFIG ────────────────────────────────────────────────────────────────
DEVICE    = torch.device("cpu")
IMG_SIZE  = 224
MODEL_FN  = "models/swin_best.pth"
LABEL_MAP = {
    0: "akiec", 1: "bcc", 2: "bkl",
    3: "df",    4: "mel", 5: "nv", 6: "vasc"
}

# define which labels are considered malignant
MALIGNANT = {"akiec", "bcc", "mel"}

# ─── MODEL WRAPPER ──────────────────────────────────────────────────────────
class SwinClassifier(torch.nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, x):
        feat = self.backbone.forward_features(x)
        feat = feat.permute(0, 3, 1, 2)
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return self.head(feat)

@st.cache_resource
def load_model():
    # build exactly as at training time:
    model = timm.create_model(
        "swin_base_patch4_window7_224",
        pretrained=False,          # or True if you want imagenet pretraining
        num_classes=len(LABEL_MAP),
        
    )
    ckpt = torch.load(MODEL_FN, map_location="cpu")
    state = ckpt["model"]         # your trained weights
    model.load_state_dict(state)
    model.eval()
    return model

@st.cache_data
def preprocess(img: Image.Image):
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])
    return tfm(img).unsqueeze(0)

# ─── STREAMLIT UI ──────────────────────────────────────────────────────────
st.title(" Swin Skin Lesion Classifier")
st.write("Upload a dermoscopy image and get the top-3 lesion predictions.")

uploaded = st.file_uploader("Choose an image…", type=["png","jpg","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image", use_column_width=True)

    with st.spinner("Classifying…"):
        model = load_model()
        inp   = preprocess(img)
        with torch.no_grad():
            logits = model(inp)
            probs  = F.softmax(logits, dim=1)[0].cpu().numpy()

    # Top-3 breakdown
    st.subheader("Top-3 Predictions")
    top3 = probs.argsort()[::-1][:3]
    for idx in top3:
        st.write(f"• **{LABEL_MAP[idx]}** — {probs[idx]*100:.1f}%")

    # Check top-1 and show advice
    top1 = top3[0]
    label = LABEL_MAP[top1]
    if label in MALIGNANT:
        st.error(
            f" **{label}** is classified as malignant. "
            "Please contact a dermatologist as soon as possible."
        )
    else:
        st.success(
            f"✅ **{label}** is classified as benign. "
            "This result is not life-threatening, but if you have any concerns, consult your doctor."
        )
