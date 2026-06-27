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
from fastapi.responses import Response

from api.model import get_model, predict, compute_stats, cache_result, get_cached_result, render_slice_image
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
            volume, mask = predict(flair_arr, t1_arr, t1ce_arr, t2_arr)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    stats = compute_stats(mask)
    job_id = cache_result(volume, mask)

    return PredictionResponse(
        status="success",
        stats=TumorStats(**stats),
        mask_shape=list(mask.shape),
        job_id=job_id,
        slice_count=mask.shape[-1],
        message="Segmentation complete",
    )


@app.get("/predict/slice/{job_id}")
def get_slice_image(job_id: str, slice_idx: int = 48):
    """
    Render a single slice (FLAIR input / predicted mask / region overlay)
    for a previously computed prediction, returned as a PNG image.

    The job_id comes from a prior /predict call. Results are cached
    server-side for 30 minutes.
    """
    result = get_cached_result(job_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No cached result for this job_id — it may have expired (30 min TTL) "
                   "or never existed. Run /predict again to generate a new one."
        )

    volume, mask = result["volume"], result["mask"]
    max_slice = volume.shape[-1] - 1
    if not (0 <= slice_idx <= max_slice):
        raise HTTPException(
            status_code=400,
            detail=f"slice_idx must be between 0 and {max_slice}"
        )

    png_bytes = render_slice_image(volume, mask, slice_idx)
    return Response(content=png_bytes, media_type="image/png")
