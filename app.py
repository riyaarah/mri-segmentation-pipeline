import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import os
import tempfile

# ── Model Definition ──────────────────────────────────────────
class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, features=[32,64,128,256]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv3D(ch, f))
            self.pools.append(nn.MaxPool3d(2))
            ch = f
        self.bottleneck = DoubleConv3D(features[-1], features[-1]*2)
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = features[-1]*2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(ch, f, 2, stride=2))
            self.decoders.append(DoubleConv3D(f*2, f))
            ch = f
        self.final = nn.Conv3d(features[0], num_classes, 1)
    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for upconv, dec, skip in zip(self.upconvs, self.decoders, skips[::-1]):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.final(x)

# ── Helpers ───────────────────────────────────────────────────
def normalize(vol):
    mi, ma = vol.min(), vol.max()
    return (vol - mi) / (ma - mi) if ma != mi else vol

def crop_center(vol, size=(96,96,96)):
    x,y,z = vol.shape[-3], vol.shape[-2], vol.shape[-1]
    sx,sy,sz = (x-size[0])//2, (y-size[1])//2, (z-size[2])//2
    if vol.ndim == 4:
        return vol[:, sx:sx+size[0], sy:sy+size[1], sz:sz+size[2]]
    return vol[sx:sx+size[0], sy:sy+size[1], sz:sz+size[2]]

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = UNet3D(in_channels=4, num_classes=4, features=[32,64,128,256])
    model_path = "models/best_model_colab.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def predict(model, device, flair, t1, t1ce, t2):
    combined = np.stack([normalize(flair), normalize(t1),
                         normalize(t1ce), normalize(t2)], axis=0)
    cropped = crop_center(combined)
    x = torch.tensor(cropped[None], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(x)
        mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
    return cropped, mask

def plot_slices(volume, mask, slice_idx):
    flair_slice = volume[0, :, :, slice_idx]
    mask_slice  = mask[:, :, slice_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0e1117')

    axes[0].imshow(flair_slice, cmap='gray')
    axes[0].set_title('FLAIR Input', color='white', fontsize=13)
    axes[0].axis('off')

    cmap = plt.cm.get_cmap('jet', 4)
    axes[1].imshow(flair_slice, cmap='gray')
    axes[1].imshow(np.ma.masked_where(mask_slice==0, mask_slice),
                   cmap=cmap, alpha=0.6, vmin=0, vmax=3)
    axes[1].set_title('Predicted Tumor Mask', color='white', fontsize=13)
    axes[1].axis('off')

    axes[2].imshow(flair_slice, cmap='gray')
    axes[2].imshow(np.ma.masked_where(mask_slice!=1, mask_slice),
                   cmap='Reds', alpha=0.7, vmin=0, vmax=3)
    axes[2].imshow(np.ma.masked_where(mask_slice!=2, mask_slice),
                   cmap='Greens', alpha=0.7, vmin=0, vmax=3)
    axes[2].imshow(np.ma.masked_where(mask_slice!=3, mask_slice),
                   cmap='Blues', alpha=0.7, vmin=0, vmax=3)
    axes[2].set_title('Region Overlay', color='white', fontsize=13)
    axes[2].axis('off')

    patches = [
        mpatches.Patch(color='red',   label='Necrotic Core'),
        mpatches.Patch(color='green', label='Edema'),
        mpatches.Patch(color='blue',  label='Enhancing Tumor'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               facecolor='#0e1117', labelcolor='white', fontsize=11)
    plt.tight_layout()
    return fig

# ── UI ────────────────────────────────────────────────────────
st.set_page_config(page_title="MRI Brain Tumor Segmentation",
                   page_icon="🧠", layout="wide")

st.title("🧠 MRI Brain Tumor Segmentation")
st.markdown("**Deep Learning pipeline for automated brain tumor detection using 3D U-Net**")
st.markdown("---")

model, device = load_model()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload MRI Files")
    st.markdown("Upload all 4 MRI modalities (.nii.gz format)")
    flair_file = st.file_uploader("FLAIR",  type=["gz", "nii"])
    t1_file    = st.file_uploader("T1",     type=["gz", "nii"])
    t1ce_file  = st.file_uploader("T1ce",   type=["gz", "nii"])
    t2_file    = st.file_uploader("T2",     type=["gz", "nii"])

    st.markdown("---")
    st.subheader("Or use a sample patient")
    use_sample = st.button("🔬 Load Sample Patient", use_container_width=True)

with col2:
    if use_sample:
        sample = "data/raw/BraTS2021_00000/BraTS2021_00000"
        if os.path.exists(f"{sample}_flair.nii.gz"):
            with st.spinner("Running segmentation..."):
                flair = nib.load(f"{sample}_flair.nii.gz").get_fdata()
                t1    = nib.load(f"{sample}_t1.nii.gz").get_fdata()
                t1ce  = nib.load(f"{sample}_t1ce.nii.gz").get_fdata()
                t2    = nib.load(f"{sample}_t2.nii.gz").get_fdata()
                volume, mask = predict(model, device, flair, t1, t1ce, t2)

            st.success("Segmentation complete!")
            slice_idx = st.slider("Select brain slice", 0, 95, 48)
            fig = plot_slices(volume, mask, slice_idx)
            st.pyplot(fig)

            st.markdown("### Tumor Region Statistics")
            c1, c2, c3 = st.columns(3)
            total = mask.size
            c1.metric("Necrotic Core",   f"{(mask==1).sum()/total*100:.2f}%")
            c2.metric("Edema",           f"{(mask==2).sum()/total*100:.2f}%")
            c3.metric("Enhancing Tumor", f"{(mask==3).sum()/total*100:.2f}%")
        else:
            st.warning("Sample data not found. Please upload files manually.")

    elif all([flair_file, t1_file, t1ce_file, t2_file]):
        with st.spinner("Running segmentation..."):
            with tempfile.TemporaryDirectory() as tmp:
                def save_upload(f, name):
                    path = os.path.join(tmp, name)
                    with open(path, 'wb') as out: out.write(f.read())
                    return path
                fp  = save_upload(flair_file, "flair.nii.gz")
                tp  = save_upload(t1_file,    "t1.nii.gz")
                tcp = save_upload(t1ce_file,  "t1ce.nii.gz")
                t2p = save_upload(t2_file,    "t2.nii.gz")
                flair = nib.load(fp).get_fdata()
                t1    = nib.load(tp).get_fdata()
                t1ce  = nib.load(tcp).get_fdata()
                t2    = nib.load(t2p).get_fdata()
                volume, mask = predict(model, device, flair, t1, t1ce, t2)

        st.success("Segmentation complete!")
        slice_idx = st.slider("Select brain slice", 0, 95, 48)
        fig = plot_slices(volume, mask, slice_idx)
        st.pyplot(fig)

        c1, c2, c3 = st.columns(3)
        total = mask.size
        c1.metric("Necrotic Core",   f"{(mask==1).sum()/total*100:.2f}%")
        c2.metric("Edema",           f"{(mask==2).sum()/total*100:.2f}%")
        c3.metric("Enhancing Tumor", f"{(mask==3).sum()/total*100:.2f}%")
    else:
        st.info("👈 Upload MRI files or click 'Load Sample Patient' to begin")
        st.markdown("""
        ### How it works
        1. **Upload** 4 MRI modalities (FLAIR, T1, T1ce, T2)
        2. **Model** runs 3D U-Net segmentation
        3. **View** tumor regions highlighted by class

        ### Tumor Classes
        - 🔴 **Necrotic Core** — dead tumor tissue
        - 🟢 **Edema** — swelling around tumor
        - 🔵 **Enhancing Tumor** — active growing region
        """)

st.markdown("---")
st.markdown("*Built with PyTorch · 3D U-Net · BraTS 2021 Dataset*")