"""
FastAPI service for MRI brain tumor segmentation.

Run locally with:
    uvicorn api.main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive API docs.
"""
import tempfile
import os

import nibabel as nib
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.model import get_model, predict, compute_stats
from api.schemas import PredictionResponse, HealthResponse, TumorStats

app = FastAPI(
    title="MRI Brain Tumor Segmentation API",
    description="Upload 4 MRI modalities (FLAIR, T1, T1ce, T2) to get automated "
                 "3D U-Net tumor segmentation with region-wise statistics.",
    version="1.0.0",
)

# Allow local frontend/Streamlit to call this during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model_on_startup():
    """Load the model once when the server starts, not on every request."""
    try:
        get_model()
        print("✓ Model loaded successfully")
    except FileNotFoundError as e:
        print(f"⚠ Warning: {e}")
        print("⚠ Server starting without model — /predict will fail until model is available")


@app.get("/", response_model=HealthResponse)
def health_check():
    """Simple health check — confirms the API is up and whether the model loaded."""
    try:
        get_model()
        return HealthResponse(status="ok", model_loaded=True)
    except FileNotFoundError:
        return HealthResponse(status="degraded", model_loaded=False)


def _load_nifti_from_upload(upload: UploadFile, tmp_dir: str) -> np.ndarray:
    """Save an uploaded file to disk temporarily and load it as a numpy array via nibabel."""
    path = os.path.join(tmp_dir, upload.filename)
    with open(path, "wb") as f:
        f.write(upload.file.read())
    return nib.load(path).get_fdata()


@app.post("/predict", response_model=PredictionResponse)
async def predict_segmentation(
    flair: UploadFile = File(..., description="FLAIR modality (.nii.gz)"),
    t1: UploadFile = File(..., description="T1 modality (.nii.gz)"),
    t1ce: UploadFile = File(..., description="T1ce modality (.nii.gz)"),
    t2: UploadFile = File(..., description="T2 modality (.nii.gz)"),
):
    """
    Run brain tumor segmentation on 4 uploaded MRI modalities.

    Returns tumor region statistics (necrotic core, edema, enhancing tumor)
    as percentages of total brain volume.
    """
    try:
        get_model()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not available on server")

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            flair_arr = _load_nifti_from_upload(flair, tmp_dir)
            t1_arr = _load_nifti_from_upload(t1, tmp_dir)
            t1ce_arr = _load_nifti_from_upload(t1ce, tmp_dir)
            t2_arr = _load_nifti_from_upload(t2, tmp_dir)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read MRI file: {e}")

        try:
            _, mask = predict(flair_arr, t1_arr, t1ce_arr, t2_arr)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    stats = compute_stats(mask)

    return PredictionResponse(
        status="success",
        stats=TumorStats(**stats),
        mask_shape=list(mask.shape),
        message="Segmentation complete",
    )
