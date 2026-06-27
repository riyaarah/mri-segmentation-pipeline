"""
Streamlit frontend for MRI Brain Tumor Segmentation.

This is a thin client — all model logic (preprocessing, inference,
visualization) lives in the deployed FastAPI service. This file just
uploads files, displays returned stats, and fetches rendered slice
images on demand as the user moves the slider.

Configure which API this talks to via the MRI_API_URL environment
variable. Defaults to the deployed HF Spaces API.
"""
import os
import streamlit as st
import requests

API_URL = os.environ.get(
    "MRI_API_URL",
    "https://riyaarahim-mri-tumor-segmentation-api.hf.space"
)

st.set_page_config(page_title="MRI Brain Tumor Segmentation",
                   page_icon="🧠", layout="wide")

st.title("🧠 MRI Brain Tumor Segmentation")
st.markdown("**Deep Learning pipeline for automated brain tumor detection using 3D U-Net**")
st.caption(f"Connected to API: `{API_URL}`")
st.markdown("---")


def call_predict(files: dict) -> dict:
    """POST the 4 MRI files to the API's /predict endpoint."""
    response = requests.post(f"{API_URL}/predict", files=files, timeout=120)
    response.raise_for_status()
    return response.json()


def fetch_slice_image(job_id: str, slice_idx: int) -> bytes:
    """GET a rendered slice image for a given job + slice index."""
    response = requests.get(
        f"{API_URL}/predict/slice/{job_id}",
        params={"slice_idx": slice_idx},
        timeout=30,
    )
    response.raise_for_status()
    return response.content


def show_results(result: dict):
    st.success(result.get("message", "Segmentation complete!"))

    job_id = result["job_id"]
    slice_count = result["slice_count"]

    slice_idx = st.slider("Select brain slice", 0, slice_count - 1, slice_count // 2)

    with st.spinner("Rendering slice..."):
        image_bytes = fetch_slice_image(job_id, slice_idx)
    st.image(image_bytes, use_container_width=True)

    stats = result["stats"]
    st.markdown("### Tumor Region Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Necrotic Core",   f"{stats['necrotic_core_pct']}%")
    c2.metric("Edema",           f"{stats['edema_pct']}%")
    c3.metric("Enhancing Tumor", f"{stats['enhancing_tumor_pct']}%")


col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload MRI Files")
    st.markdown("Upload all 4 MRI modalities (.nii.gz format)")
    flair_file = st.file_uploader("FLAIR",  type=["gz", "nii"])
    t1_file    = st.file_uploader("T1",     type=["gz", "nii"])
    t1ce_file  = st.file_uploader("T1ce",   type=["gz", "nii"])
    t2_file    = st.file_uploader("T2",     type=["gz", "nii"])

    run_clicked = st.button(
        "🔬 Run Segmentation",
        use_container_width=True,
        disabled=not all([flair_file, t1_file, t1ce_file, t2_file]),
    )

with col2:
    if run_clicked:
        with st.spinner("Running segmentation (this calls the live API)..."):
            try:
                files = {
                    "flair": (flair_file.name, flair_file.getvalue()),
                    "t1": (t1_file.name, t1_file.getvalue()),
                    "t1ce": (t1ce_file.name, t1ce_file.getvalue()),
                    "t2": (t2_file.name, t2_file.getvalue()),
                }
                result = call_predict(files)
                st.session_state["last_result"] = result
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
                st.stop()

    if "last_result" in st.session_state:
        show_results(st.session_state["last_result"])
    elif not run_clicked:
        st.info("👈 Upload all 4 MRI files, then click 'Run Segmentation'")
        st.markdown("""
        ### How it works
        1. **Upload** 4 MRI modalities (FLAIR, T1, T1ce, T2)
        2. This app sends them to a **live FastAPI service** running the 3D U-Net model
        3. **View** tumor regions highlighted by class, rendered by the API

        ### Tumor Classes
        - 🔴 **Necrotic Core** — dead tumor tissue
        - 🟢 **Edema** — swelling around tumor
        - 🔵 **Enhancing Tumor** — active growing region

        ### Architecture
        This frontend is a thin client. All model inference and image
        rendering happens server-side via a separately deployed
        FastAPI service — see the `/docs` endpoint of the API for
        its full interface.
        """)

st.markdown("---")
st.markdown("*Frontend (Streamlit) → API (FastAPI) → Model (3D U-Net, BraTS 2021)*")
