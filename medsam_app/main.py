"""
MedSAM2 Web Interface — FastAPI backend
Port 9000
"""
import os
import sys
import uuid
import contextlib
import json
import shutil
import zipfile
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple

# Add MedSAM2 root to Python path
MEDSAM2_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(MEDSAM2_ROOT))

import numpy as np
import torch
from PIL import Image
import io
import base64

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# Optional heavy imports — fail gracefully so the app still starts
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    from totalsegmentator.python_api import totalsegmentator as _totalsegmentator
    from totalsegmentator.map_to_binary import class_map as TOTALSEG_CLASS_MAP
    HAS_TOTALSEG = True
except ImportError:
    HAS_TOTALSEG = False
    TOTALSEG_CLASS_MAP = {}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="MedSAM2 Web Interface")

UPLOAD_DIR = Path("/tmp/medsam2_uploads")
RESULT_DIR = Path("/tmp/medsam2_results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory stores
uploaded_files: dict = {}   # file_id → {path, shape, spacing}
seg_jobs: dict = {}         # job_id → {status, result_path, error, progress}
loaded_model: dict = {}     # {checkpoint, predictor, device}

CHECKPOINT_DIR = MEDSAM2_ROOT / "checkpoints"
CONFIG_DIR = MEDSAM2_ROOT / "sam2" / "configs"

CHECKPOINT_TO_CONFIG = {
    "MedSAM2_latest.pt": "sam2.1_hiera_t512.yaml",
    "MedSAM2_CTLesion.pt": "sam2.1_hiera_t512.yaml",
    "MedSAM2_MRI_LiverLesion.pt": "sam2.1_hiera_t512.yaml",
    "MedSAM2_US_Heart.pt": "sam2.1_hiera_t512.yaml",
    "MedSAM2_2411.pt": "sam2.1_hiera_t512.yaml",
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_nifti(path: str):
    """Load a NIfTI file and return (array, spacing). Array shape: (D, H, W)."""
    if not HAS_SITK:
        raise RuntimeError("SimpleITK not installed. Run: pip install SimpleITK")
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (D, H, W) for axial
    spacing = img.GetSpacing()  # (x, y, z) → (W, H, D) in ITK convention
    return arr, spacing, img


# ---------------------------------------------------------------------------
# DICOM helpers
# ---------------------------------------------------------------------------

def collect_dcm_files(directory: str) -> List[str]:
    """Recursively find all .dcm files under a directory."""
    dcm_files = []
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith('.dcm') or f.lower().endswith('.ima'):
                dcm_files.append(os.path.join(root, f))
            # also accept DICOM files with no extension (common in some PACS exports)
            elif '.' not in f:
                dcm_files.append(os.path.join(root, f))
    return dcm_files


def read_dicom_tag(path: str, tag: str) -> str:
    """Safely read a DICOM metadata tag via SimpleITK."""
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        return reader.GetMetaData(tag) if reader.HasMetaDataKey(tag) else ""
    except Exception:
        return ""


def load_dicom_series(dicom_dir: str) -> Tuple[sitk.Image, dict]:
    """
    Read a DICOM series from a directory using SimpleITK.
    Returns (sitk_image, metadata_dict).

    Strategy:
      1. Use GDCM series reader (works for all proper clinical DICOMs).
      2. Fallback: if best series has only 1 slice but many DCM files exist,
         sort all files by instance number and stack manually — handles
         DICOMs from scanners that don't write consistent Series UIDs.
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise ValueError("No DICOM series found. Ensure the files are valid DICOM (.dcm).")

    # Pick series with most slices
    series_file_map = {sid: reader.GetGDCMSeriesFileNames(dicom_dir, sid) for sid in series_ids}
    best_id = max(series_file_map, key=lambda sid: len(series_file_map[sid]))
    dicom_names = list(series_file_map[best_id])

    # Fallback: if best series has 1 file but multiple DCM files exist, use all DCM files
    all_dcm = collect_dcm_files(dicom_dir)
    if len(dicom_names) == 1 and len(all_dcm) > 1:
        dicom_names = _sort_dcm_by_instance(all_dcm)

    if not dicom_names:
        raise ValueError("Could not retrieve file list for DICOM series.")

    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()

    # Pull useful metadata from the first slice
    meta = {}
    try:
        for tag, label in [
            ("0008|0060", "modality"),
            ("0008|103e", "series_description"),
            ("0010|0010", "patient_name"),
            ("0008|0020", "study_date"),
            ("0018|0050", "slice_thickness"),
        ]:
            if reader.HasMetaDataKey(0, tag):
                meta[label] = reader.GetMetaData(0, tag).strip()
    except Exception:
        pass

    return image, meta


def _sort_dcm_by_instance(dcm_files: List[str]) -> List[str]:
    """Sort DICOM files by Instance Number tag (0020,0013), falling back to filename."""
    def sort_key(path):
        try:
            r = sitk.ImageFileReader()
            r.SetFileName(path)
            r.ReadImageInformation()
            if r.HasMetaDataKey("0020|0013"):
                return int(r.GetMetaData("0020|0013").strip())
        except Exception:
            pass
        return os.path.basename(path)
    return sorted(dcm_files, key=sort_key)


def dicom_to_nifti(dicom_dir: str, out_path: str) -> Tuple[sitk.Image, dict]:
    """Convert a DICOM series directory to a NIfTI file. Returns (sitk_image, metadata)."""
    image, meta = load_dicom_series(dicom_dir)
    sitk.WriteImage(image, out_path)
    return image, meta


def apply_window(arr: np.ndarray, wc: float, ww: float) -> np.ndarray:
    """Apply CT window/level and normalise to uint8 [0, 255]."""
    low = wc - ww / 2
    high = wc + ww / 2
    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low) * 255.0
    return arr.astype(np.uint8)


def auto_window(arr: np.ndarray) -> Tuple[float, float]:
    """Auto window: 1st–99th percentile."""
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    ww = p99 - p1
    wc = (p99 + p1) / 2
    return float(wc), float(ww)


def slice_to_png(slice_2d: np.ndarray) -> bytes:
    """Convert a uint8 (H, W) slice to PNG bytes."""
    img = Image.fromarray(slice_2d, mode="L").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def overlay_mask_on_slice(slice_2d: np.ndarray, mask_2d: np.ndarray, alpha: float = 0.45) -> bytes:
    """Return a transparent RGBA PNG: colored only where the mask is non-zero."""
    H, W = slice_2d.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    if mask_2d.any():
        rgba[mask_2d > 0] = [255, 230, 30, 255]
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def resize_to_rgb_512(arr_3d: np.ndarray) -> np.ndarray:
    """
    arr_3d: (D, H, W) uint8
    returns: (D, 3, 512, 512) float32 normalised with ImageNet stats
    """
    d, h, w = arr_3d.shape
    out = np.zeros((d, 3, 512, 512), dtype=np.float32)
    for i in range(d):
        img = Image.fromarray(arr_3d[i]).convert("RGB").resize((512, 512), Image.BILINEAR)
        arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
        out[i] = arr
    return out


def get_largest_cc(seg: np.ndarray) -> np.ndarray:
    if not HAS_SKIMAGE:
        return seg
    labels = measure.label(seg)
    if labels.max() == 0:
        return seg
    counts = np.bincount(labels.flat)[1:]
    largest = counts.argmax() + 1
    return (labels == largest).astype(np.uint8)


# ---------------------------------------------------------------------------
# TotalSegmentator constants
# ---------------------------------------------------------------------------

# Structures grouped for the UI, per task
TOTALSEG_GROUPS = {
    "total_mr": {
        "Abdomen": ["liver", "spleen", "gallbladder", "pancreas", "stomach",
                    "duodenum", "small_bowel", "colon"],
        "Urogenital": ["kidney_right", "kidney_left", "urinary_bladder",
                       "prostate", "adrenal_gland_right", "adrenal_gland_left"],
        "Cardiovascular": ["heart", "aorta", "inferior_vena_cava",
                           "portal_vein_and_splenic_vein", "iliac_artery_left",
                           "iliac_artery_right", "iliac_vena_left", "iliac_vena_right"],
        "Spine / MSK": ["vertebrae", "intervertebral_discs", "spinal_cord",
                        "sacrum", "femur_left", "femur_right", "hip_left", "hip_right",
                        "humerus_left", "humerus_right"],
        "Other": ["lung_left", "lung_right", "esophagus", "brain"],
    },
    "total": {
        "Abdomen": ["liver", "spleen", "gallbladder", "pancreas", "stomach",
                    "duodenum", "small_bowel", "colon"],
        "Urogenital": ["kidney_right", "kidney_left", "urinary_bladder",
                       "prostate", "adrenal_gland_right", "adrenal_gland_left",
                       "kidney_cyst_left", "kidney_cyst_right"],
        "Cardiovascular": ["heart", "aorta", "inferior_vena_cava",
                           "portal_vein_and_splenic_vein", "pulmonary_vein",
                           "superior_vena_cava", "iliac_artery_left", "iliac_artery_right"],
        "Thorax": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                   "lung_middle_lobe_right", "lung_lower_lobe_right",
                   "esophagus", "trachea", "thyroid_gland"],
        "Spine / MSK": ["sacrum", "vertebrae_L5", "vertebrae_L4", "vertebrae_L3",
                        "vertebrae_L2", "vertebrae_L1", "spinal_cord",
                        "femur_left", "femur_right", "hip_left", "hip_right"],
    },
}

# Per-structure overlay colours (RGB)
STRUCTURE_COLORS: dict = {
    "pancreas":                  [255, 200,   0],
    "liver":                     [100, 200, 100],
    "spleen":                    [100, 150, 255],
    "kidney_right":              [255, 100, 100],
    "kidney_left":               [255, 130, 130],
    "gallbladder":               [180, 255, 100],
    "stomach":                   [255, 160,  80],
    "duodenum":                  [200, 100, 255],
    "small_bowel":               [255, 220, 150],
    "colon":                     [200, 160, 100],
    "adrenal_gland_right":       [255,  80, 200],
    "adrenal_gland_left":        [255, 120, 220],
    "urinary_bladder":           [ 80, 220, 255],
    "prostate":                  [255,  80,  80],
    "heart":                     [255,  60,  60],
    "aorta":                     [255, 100, 100],
    "lung_left":                 [150, 200, 255],
    "lung_right":                [150, 200, 255],
    "esophagus":                 [220, 180, 255],
    "spinal_cord":               [255, 255, 100],
    "_default":                  [200, 200, 200],
}


def structure_color(name: str) -> List[int]:
    return STRUCTURE_COLORS.get(name, STRUCTURE_COLORS["_default"])


def overlay_multilabel_on_slice(slice_2d: np.ndarray, mask_2d: np.ndarray,
                                 label_map: dict, alpha: float = 0.5) -> bytes:
    """
    label_map: {label_int: structure_name}
    Returns a transparent RGBA PNG: each structure is its colour, transparent elsewhere.
    """
    H, W = slice_2d.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for label_id, name in label_map.items():
        region = mask_2d == int(label_id)
        if not region.any():
            continue
        color = structure_color(name)
        rgba[region, :3] = color
        rgba[region, 3] = 255  # fully opaque where mask exists
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _blend_slice(sl: np.ndarray, mk: np.ndarray, label_map: Optional[dict]) -> np.ndarray:
    """Return an RGB uint8 array for a 2D slice+mask, without writing to bytes."""
    if label_map:
        rgb = np.stack([sl, sl, sl], axis=-1).astype(np.float32)
        for label_id, name in label_map.items():
            region = mk == int(label_id)
            if not region.any():
                continue
            color = np.array(structure_color(name), dtype=np.float32)
            rgb[region] = rgb[region] * 0.5 + color * 0.5
        return rgb.clip(0, 255).astype(np.uint8)
    else:
        rgb = np.stack([sl, sl, sl], axis=-1).astype(np.uint8)
        if mk.any():
            overlay = np.zeros_like(rgb)
            overlay[mk > 0] = [255, 230, 30]
            rgb = np.where(mk[:, :, None] > 0,
                           (rgb.astype(np.float32) * 0.55 + overlay.astype(np.float32) * 0.45).astype(np.uint8),
                           rgb)
        return rgb


def render_mpr_slice(img_w: np.ndarray, mask: np.ndarray, plane: str,
                     slice_idx: int, label_map: Optional[dict],
                     spacing: List[float], target_w: int = 290,
                     axial_pos: Optional[int] = None,
                     coronal_pos: Optional[int] = None,
                     sagittal_pos: Optional[int] = None) -> bytes:
    """
    Render a coronal or sagittal MPR slice with overlay and crosshair lines.

    spacing: SimpleITK convention (sx, sy, sz) = (col, row, slice) spacing in mm.
    axial_pos:    draw a horizontal line at this axial slice index.
    coronal_pos:  draw a vertical line at this coronal y-index (sagittal view only).
    sagittal_pos: draw a vertical line at this sagittal x-index (coronal view only).
    """
    from PIL import ImageDraw
    D, H, W = img_w.shape
    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])

    if plane == "coronal":
        slice_idx = max(0, min(slice_idx, H - 1))
        sl = img_w[:, slice_idx, :]   # (D, W) — rows=superior-inferior, cols=left-right
        mk = mask[:, slice_idx, :]
        phys_w = W * sx
        phys_h = D * sz
    else:  # sagittal
        slice_idx = max(0, min(slice_idx, W - 1))
        sl = img_w[:, :, slice_idx]   # (D, H) — rows=superior-inferior, cols=anterior-posterior
        mk = mask[:, :, slice_idx]
        phys_w = H * sy
        phys_h = D * sz

    # Blend overlay
    rgb = _blend_slice(sl, mk, label_map)

    # Resize preserving physical aspect ratio
    aspect = phys_w / phys_h if phys_h > 0 else 1.0
    target_h = max(1, int(target_w / aspect))
    pil = Image.fromarray(rgb).resize((target_w, target_h), Image.BILINEAR)

    # Draw crosshair lines
    draw = ImageDraw.Draw(pil)

    def _line_y(pos, total):
        return max(0, min(int(pos / max(total, 1) * target_h), target_h - 1))

    def _line_x(pos, total):
        return max(0, min(int(pos / max(total, 1) * target_w), target_w - 1))

    if axial_pos is not None:
        ly = _line_y(axial_pos, D)
        draw.line([(0, ly), (target_w, ly)], fill=(255, 50, 50), width=1)

    if plane == "coronal" and sagittal_pos is not None:
        lx = _line_x(sagittal_pos, W)
        draw.line([(lx, 0), (lx, target_h)], fill=(50, 220, 255), width=1)

    if plane == "sagittal" and coronal_pos is not None:
        lx = _line_x(coronal_pos, H)
        draw.line([(lx, 0), (lx, target_h)], fill=(50, 255, 100), width=1)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_predictor(checkpoint_name: str):
    from sam2.build_sam import build_sam2_video_predictor_npz
    ckpt_path = str(CHECKPOINT_DIR / checkpoint_name)
    cfg_name = CHECKPOINT_TO_CONFIG.get(checkpoint_name, "sam2.1_hiera_t512.yaml")
    cfg_path = f"configs/{cfg_name}"  # relative, SAM2 uses Hydra config resolution
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    device = get_device()
    predictor = build_sam2_video_predictor_npz(cfg_path, ckpt_path, device=device)
    return predictor, device


def get_predictor(checkpoint_name: str):
    if loaded_model.get("checkpoint") != checkpoint_name:
        print(f"Loading model: {checkpoint_name}...")
        predictor, device = load_predictor(checkpoint_name)
        loaded_model["checkpoint"] = checkpoint_name
        loaded_model["predictor"] = predictor
        loaded_model["device"] = device
        print(f"Model loaded on {device}")
    return loaded_model["predictor"], loaded_model["device"]


# ---------------------------------------------------------------------------
# Segmentation worker
# ---------------------------------------------------------------------------

def run_segmentation(
    job_id: str,
    file_id: str,
    key_slice: int,
    bbox: list,         # [x0, y0, x1, y1] in original image coords
    checkpoint: str,
    wc: float,
    ww: float,
):
    try:
        seg_jobs[job_id]["status"] = "running"
        seg_jobs[job_id]["progress"] = 5

        file_info = uploaded_files[file_id]
        arr, spacing, sitk_img = load_nifti(file_info["path"])
        seg_jobs[job_id]["progress"] = 15

        # Windowed uint8
        arr_w = apply_window(arr, wc, ww)
        D, H, W = arr_w.shape

        # Resize to (D, 3, 512, 512) and normalise
        img_resized = resize_to_rgb_512(arr_w)
        seg_jobs[job_id]["progress"] = 30

        device_str = get_device()
        img_tensor = torch.from_numpy(img_resized).to(device_str)
        img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None].to(device_str)
        img_std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None].to(device_str)
        img_tensor = img_tensor - img_mean
        img_tensor = img_tensor / img_std
        seg_jobs[job_id]["progress"] = 40

        # Scale bbox from original image coords to 512x512
        scale_x = 512.0 / W
        scale_y = 512.0 / H
        bbox_scaled = np.array([
            bbox[0] * scale_x,
            bbox[1] * scale_y,
            bbox[2] * scale_x,
            bbox[3] * scale_y,
        ])

        predictor, device = get_predictor(checkpoint)
        seg_jobs[job_id]["progress"] = 55

        segs_3D = np.zeros((D, H, W), dtype=np.uint8)

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device_str == "cuda"
            else contextlib.nullcontext()
        )

        with torch.inference_mode(), autocast_ctx:
            inference_state = predictor.init_state(img_tensor, H, W)

            # Forward propagation
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=key_slice,
                obj_id=1,
                box=bbox_scaled,
            )
            for out_frame_idx, _out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                segs_3D[out_frame_idx][
                    (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                ] = 1

            predictor.reset_state(inference_state)

            # Backward propagation
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=key_slice,
                obj_id=1,
                box=bbox_scaled,
            )
            for out_frame_idx, _out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state, reverse=True
            ):
                segs_3D[out_frame_idx][
                    (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                ] = 1

            predictor.reset_state(inference_state)

        seg_jobs[job_id]["progress"] = 85

        if segs_3D.max() > 0:
            segs_3D = get_largest_cc(segs_3D)

        # Save result
        result_dir = RESULT_DIR / job_id
        result_dir.mkdir(parents=True, exist_ok=True)

        # Windowed image NIfTI
        sitk_img_w = sitk.GetImageFromArray(arr_w)
        sitk_img_w.CopyInformation(sitk_img)
        sitk.WriteImage(sitk_img_w, str(result_dir / "image_windowed.nii.gz"))

        # Mask NIfTI
        sitk_mask = sitk.GetImageFromArray(segs_3D)
        sitk_mask.CopyInformation(sitk_img)
        sitk.WriteImage(sitk_mask, str(result_dir / "segmentation.nii.gz"))

        # Cache numpy for fast overlay rendering
        np.save(str(result_dir / "mask.npy"), segs_3D)
        np.save(str(result_dir / "image_w.npy"), arr_w)

        seg_jobs[job_id]["status"] = "done"
        seg_jobs[job_id]["progress"] = 100
        seg_jobs[job_id]["result_dir"] = str(result_dir)
        seg_jobs[job_id]["num_slices"] = D
        seg_jobs[job_id]["voxels"] = int(segs_3D.sum())

    except Exception as exc:
        import traceback
        seg_jobs[job_id]["status"] = "error"
        seg_jobs[job_id]["error"] = str(exc)
        seg_jobs[job_id]["traceback"] = traceback.format_exc()
        print(f"[ERROR] Job {job_id}: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# TotalSegmentator worker
# ---------------------------------------------------------------------------

def run_totalseg_segmentation(
    job_id: str,
    file_id: str,
    task: str,
    structures: List[str],
    wc: float,
    ww: float,
):
    try:
        seg_jobs[job_id]["status"] = "running"
        seg_jobs[job_id]["progress"] = 5

        file_info = uploaded_files[file_id]
        arr, spacing, sitk_img = load_nifti(file_info["path"])
        arr_w = apply_window(arr, wc, ww)
        D, H, W = arr_w.shape
        seg_jobs[job_id]["progress"] = 15

        result_dir = RESULT_DIR / job_id
        result_dir.mkdir(parents=True, exist_ok=True)
        ts_out_dir = result_dir / "totalseg_out"
        ts_out_dir.mkdir(exist_ok=True)

        # Map structure names to label ids for this task
        class_map = TOTALSEG_CLASS_MAP.get(task, {})
        # Build name→id lookup (only for requested structures)
        name_to_id = {v: k for k, v in class_map.items()}
        valid_structures = [s for s in structures if s in name_to_id]
        if not valid_structures:
            raise ValueError(f"None of {structures} found in task '{task}'")

        seg_jobs[job_id]["progress"] = 20
        seg_jobs[job_id]["status_detail"] = "Running TotalSegmentator…"

        # Map get_device() → totalsegmentator device string
        _dev = get_device()
        ts_device = {"cuda": "gpu", "mps": "mps", "cpu": "cpu"}.get(_dev, "cpu")

        _totalsegmentator(
            input=file_info["path"],
            output=str(ts_out_dir),
            task=task,
            roi_subset=valid_structures,
            device=ts_device,
            quiet=True,
            verbose=False,
        )

        seg_jobs[job_id]["progress"] = 80

        # Combine per-structure masks into one multi-label array
        combined = np.zeros((D, H, W), dtype=np.uint8)
        label_map = {}   # {label_id: structure_name}
        found_structures = []

        for struct in valid_structures:
            mask_path = ts_out_dir / f"{struct}.nii.gz"
            if not mask_path.exists():
                continue
            mask_sitk = sitk.ReadImage(str(mask_path))
            mask_arr = sitk.GetArrayFromImage(mask_sitk)
            # Resize if needed (shouldn't be, but guard anyway)
            if mask_arr.shape != (D, H, W):
                mask_arr = np.where(
                    np.array(Image.fromarray(mask_arr.astype(np.uint8)).resize(
                        (W, H), Image.NEAREST)) > 0, 1, 0
                ).astype(np.uint8)
            label_id = name_to_id[struct]
            combined[mask_arr > 0] = label_id
            label_map[str(label_id)] = struct
            found_structures.append(struct)

        seg_jobs[job_id]["progress"] = 90

        # Save results
        np.save(str(result_dir / "mask.npy"), combined)
        np.save(str(result_dir / "image_w.npy"), arr_w)
        with open(str(result_dir / "label_map.json"), "w") as f:
            json.dump(label_map, f)

        # Save combined NIfTI mask
        sitk_mask = sitk.GetImageFromArray(combined)
        sitk_mask.CopyInformation(sitk_img)
        sitk.WriteImage(sitk_mask, str(result_dir / "segmentation.nii.gz"))

        # Save windowed image NIfTI
        sitk_img_w = sitk.GetImageFromArray(arr_w)
        sitk_img_w.CopyInformation(sitk_img)
        sitk.WriteImage(sitk_img_w, str(result_dir / "image_windowed.nii.gz"))

        seg_jobs[job_id]["status"] = "done"
        seg_jobs[job_id]["progress"] = 100
        seg_jobs[job_id]["result_dir"] = str(result_dir)
        seg_jobs[job_id]["num_slices"] = D
        seg_jobs[job_id]["voxels"] = int((combined > 0).sum())
        seg_jobs[job_id]["mask_type"] = "multilabel"
        seg_jobs[job_id]["label_map"] = label_map
        seg_jobs[job_id]["structures"] = found_structures

    except Exception as exc:
        import traceback
        seg_jobs[job_id]["status"] = "error"
        seg_jobs[job_id]["error"] = str(exc)
        seg_jobs[job_id]["traceback"] = traceback.format_exc()
        print(f"[TotalSeg ERROR] Job {job_id}: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/checkpoints")
async def list_checkpoints():
    if not CHECKPOINT_DIR.exists():
        return {"checkpoints": [], "dir": str(CHECKPOINT_DIR)}
    ckpts = [f.name for f in CHECKPOINT_DIR.glob("*.pt")]
    return {"checkpoints": sorted(ckpts), "dir": str(CHECKPOINT_DIR)}


@app.post("/api/upload")
async def upload_file(files: List[UploadFile] = File(...)):
    """
    Accept one of:
      - A single .nii / .nii.gz file
      - A single .zip containing a DICOM series
      - One or more .dcm / .ima files (the full series)
    Converts DICOM to NIfTI automatically.
    """
    if not HAS_SITK:
        raise HTTPException(503, "SimpleITK not installed. Run: pip install SimpleITK")
    if not files:
        raise HTTPException(400, "No files provided")

    file_id = str(uuid.uuid4())
    work_dir = UPLOAD_DIR / file_id
    work_dir.mkdir(parents=True, exist_ok=True)

    first_name = files[0].filename or ""
    suffix = "".join(Path(first_name).suffixes).lower()
    dicom_meta = {}

    # ── NIfTI ────────────────────────────────────────────────────────────────
    if suffix in (".nii.gz", ".nii"):
        save_path = work_dir / f"image{suffix}"
        with open(save_path, "wb") as f:
            shutil.copyfileobj(files[0].file, f)
        display_name = first_name
        source_type = "nifti"

    # ── ZIP (DICOM series) ────────────────────────────────────────────────────
    elif suffix == ".zip":
        zip_path = work_dir / "upload.zip"
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(files[0].file, f)
        extract_dir = work_dir / "dicom"
        extract_dir.mkdir()
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(extract_dir))
        save_path = work_dir / "image.nii.gz"
        _, dicom_meta = dicom_to_nifti(str(extract_dir), str(save_path))
        display_name = first_name.replace(".zip", "")
        source_type = "dicom_zip"

    # ── Raw DICOM files (.dcm / .ima / no extension) ─────────────────────────
    else:
        dcm_dir = work_dir / "dicom"
        dcm_dir.mkdir()
        for uf in files:
            # webkitdirectory sends relative paths like "folder/sub/file.dcm" — flatten to basename
            raw_name = uf.filename or f"slice_{uuid.uuid4().hex[:8]}.dcm"
            safe_name = Path(raw_name).name or f"slice_{uuid.uuid4().hex[:8]}.dcm"
            dest = dcm_dir / safe_name
            with open(dest, "wb") as f:
                shutil.copyfileobj(uf.file, f)
        save_path = work_dir / "image.nii.gz"
        _, dicom_meta = dicom_to_nifti(str(dcm_dir), str(save_path))
        display_name = dicom_meta.get("series_description") or Path(first_name).stem
        source_type = "dicom_files"

    arr, spacing, _ = load_nifti(str(save_path))
    wc, ww = auto_window(arr)

    uploaded_files[file_id] = {
        "path": str(save_path),
        "shape": list(arr.shape),
        "spacing": list(spacing),
        "filename": display_name,
        "auto_wc": wc,
        "auto_ww": ww,
        "source_type": source_type,
        "dicom_meta": dicom_meta,
    }

    return {
        "file_id": file_id,
        "filename": display_name,
        "shape": arr.shape,
        "num_slices": arr.shape[0],
        "auto_wc": round(wc, 1),
        "auto_ww": round(ww, 1),
        "source_type": source_type,
        "modality": dicom_meta.get("modality", ""),
        "series_description": dicom_meta.get("series_description", ""),
    }


@app.get("/api/slice/{file_id}/{slice_idx}")
async def get_slice(file_id: str, slice_idx: int, wc: float = None, ww: float = None):
    if file_id not in uploaded_files:
        raise HTTPException(404, "File not found")
    info = uploaded_files[file_id]
    arr, _, _ = load_nifti(info["path"])
    D = arr.shape[0]
    slice_idx = max(0, min(slice_idx, D - 1))

    if wc is None:
        wc = info["auto_wc"]
    if ww is None:
        ww = info["auto_ww"]

    arr_w = apply_window(arr, wc, ww)
    png_bytes = slice_to_png(arr_w[slice_idx])
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


class SegmentRequest(BaseModel):
    file_id: str
    key_slice: int
    bbox: list          # [x0, y0, x1, y1]
    checkpoint: str
    wc: Optional[float] = None
    ww: Optional[float] = None


@app.post("/api/segment")
async def start_segmentation(req: SegmentRequest, background_tasks: BackgroundTasks):
    if req.file_id not in uploaded_files:
        raise HTTPException(404, "File not found")

    info = uploaded_files[req.file_id]
    wc = req.wc if req.wc is not None else info["auto_wc"]
    ww = req.ww if req.ww is not None else info["auto_ww"]

    job_id = str(uuid.uuid4())
    seg_jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "file_id": req.file_id,
        "result_dir": None,
        "error": None,
    }

    background_tasks.add_task(
        run_segmentation,
        job_id=job_id,
        file_id=req.file_id,
        key_slice=req.key_slice,
        bbox=req.bbox,
        checkpoint=req.checkpoint,
        wc=wc,
        ww=ww,
    )

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in seg_jobs:
        raise HTTPException(404, "Job not found")
    job = seg_jobs[job_id]
    return {
        "status": job["status"],
        "progress": job.get("progress", 0),
        "error": job.get("error"),
        "num_slices": job.get("num_slices"),
        "voxels": job.get("voxels"),
        "mask_type": job.get("mask_type", "binary"),
        "label_map": job.get("label_map", {}),
        "structures": job.get("structures", []),
    }


@app.get("/api/totalseg/structures/{task}")
async def get_totalseg_structures(task: str):
    if not HAS_TOTALSEG:
        raise HTTPException(503, "TotalSegmentator not installed")
    groups = TOTALSEG_GROUPS.get(task)
    if groups is None:
        raise HTTPException(400, f"Unknown task: {task}. Use 'total' or 'total_mr'")
    # Filter to structures that actually exist in this task's class map
    class_map = TOTALSEG_CLASS_MAP.get(task, {})
    valid = set(class_map.values())
    filtered = {g: [s for s in structs if s in valid] for g, structs in groups.items()}
    return {"task": task, "groups": filtered}


class TotalsegRequest(BaseModel):
    file_id: str
    task: str = "total_mr"
    structures: List[str] = ["pancreas"]
    wc: Optional[float] = None
    ww: Optional[float] = None


@app.post("/api/segment/totalseg")
async def start_totalseg(req: TotalsegRequest, background_tasks: BackgroundTasks):
    if not HAS_TOTALSEG:
        raise HTTPException(503, "TotalSegmentator not installed. Run: pip install TotalSegmentator")
    if req.file_id not in uploaded_files:
        raise HTTPException(404, "File not found")

    info = uploaded_files[req.file_id]
    wc = req.wc if req.wc is not None else info["auto_wc"]
    ww = req.ww if req.ww is not None else info["auto_ww"]

    job_id = str(uuid.uuid4())
    seg_jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "file_id": req.file_id,
        "result_dir": None,
        "error": None,
        "mask_type": "multilabel",
        "label_map": {},
    }

    background_tasks.add_task(
        run_totalseg_segmentation,
        job_id=job_id,
        file_id=req.file_id,
        task=req.task,
        structures=req.structures,
        wc=wc,
        ww=ww,
    )
    return {"job_id": job_id}


@app.get("/api/result/slice/{job_id}/{slice_idx}")
async def get_result_slice(job_id: str, slice_idx: int):
    if job_id not in seg_jobs:
        raise HTTPException(404, "Job not found")
    job = seg_jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, "Segmentation not complete")

    result_dir = Path(job["result_dir"])
    mask = np.load(str(result_dir / "mask.npy"))
    img_w = np.load(str(result_dir / "image_w.npy"))

    D = mask.shape[0]
    slice_idx = max(0, min(slice_idx, D - 1))

    if job.get("mask_type") == "multilabel":
        label_map = job.get("label_map", {})
        png_bytes = overlay_multilabel_on_slice(img_w[slice_idx], mask[slice_idx], label_map)
    else:
        png_bytes = overlay_mask_on_slice(img_w[slice_idx], mask[slice_idx])

    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/api/download/{job_id}/mask")
async def download_mask(job_id: str):
    if job_id not in seg_jobs or seg_jobs[job_id]["status"] != "done":
        raise HTTPException(404, "Result not ready")
    result_dir = Path(seg_jobs[job_id]["result_dir"])
    mask_path = result_dir / "segmentation.nii.gz"
    return FileResponse(str(mask_path), filename="medsam2_segmentation.nii.gz",
                        media_type="application/gzip")


@app.get("/api/download/{job_id}/image")
async def download_image(job_id: str):
    if job_id not in seg_jobs or seg_jobs[job_id]["status"] != "done":
        raise HTTPException(404, "Result not ready")
    result_dir = Path(seg_jobs[job_id]["result_dir"])
    img_path = result_dir / "image_windowed.nii.gz"
    return FileResponse(str(img_path), filename="medsam2_image_windowed.nii.gz",
                        media_type="application/gzip")


@app.get("/api/mpr/{file_id}/{plane}/{slice_idx}")
async def get_raw_mpr(
    file_id: str,
    plane: str,
    slice_idx: int,
    wc: Optional[float] = None,
    ww: Optional[float] = None,
    axial_pos: Optional[int] = None,
    coronal_pos: Optional[int] = None,
    sagittal_pos: Optional[int] = None,
    target_w: int = 290,
):
    """MPR slice from raw uploaded image — no segmentation needed."""
    if file_id not in uploaded_files:
        raise HTTPException(404, "File not found")
    if plane not in ("coronal", "sagittal"):
        raise HTTPException(400, "plane must be 'coronal' or 'sagittal'")

    info = uploaded_files[file_id]
    arr, _, _ = load_nifti(info["path"])
    _wc = wc if wc is not None else info["auto_wc"]
    _ww = ww if ww is not None else info["auto_ww"]
    arr_w = apply_window(arr, _wc, _ww)

    D, H, W = arr_w.shape
    mask = np.zeros((D, H, W), dtype=np.uint8)   # empty mask — no overlay

    spacing = info.get("spacing", [1.0, 1.0, 1.0])
    png = render_mpr_slice(
        arr_w, mask, plane, slice_idx, None, spacing,
        target_w=target_w,
        axial_pos=axial_pos,
        coronal_pos=coronal_pos,
        sagittal_pos=sagittal_pos,
    )
    return StreamingResponse(io.BytesIO(png), media_type="image/png")


@app.get("/api/result/dims/{job_id}")
async def get_result_dims(job_id: str):
    if job_id not in seg_jobs or seg_jobs[job_id]["status"] != "done":
        raise HTTPException(400, "Not ready")
    result_dir = Path(seg_jobs[job_id]["result_dir"])
    mask = np.load(str(result_dir / "mask.npy"), mmap_mode="r")
    D, H, W = mask.shape
    file_id = seg_jobs[job_id].get("file_id", "")
    spacing = uploaded_files.get(file_id, {}).get("spacing", [1.0, 1.0, 1.0])
    return {"D": D, "H": H, "W": W, "spacing": spacing}


@app.get("/api/result/mpr/{job_id}/{plane}/{slice_idx}")
async def get_mpr_slice(
    job_id: str,
    plane: str,
    slice_idx: int,
    axial_pos: Optional[int] = None,
    coronal_pos: Optional[int] = None,
    sagittal_pos: Optional[int] = None,
    target_w: int = 290,
):
    if job_id not in seg_jobs or seg_jobs[job_id]["status"] != "done":
        raise HTTPException(400, "Segmentation not complete")
    if plane not in ("coronal", "sagittal"):
        raise HTTPException(400, "plane must be 'coronal' or 'sagittal'")

    job = seg_jobs[job_id]
    result_dir = Path(job["result_dir"])
    mask  = np.load(str(result_dir / "mask.npy"))
    img_w = np.load(str(result_dir / "image_w.npy"))

    label_map = job.get("label_map") if job.get("mask_type") == "multilabel" else None
    file_id   = job.get("file_id", "")
    spacing   = uploaded_files.get(file_id, {}).get("spacing", [1.0, 1.0, 1.0])

    png = render_mpr_slice(
        img_w, mask, plane, slice_idx, label_map, spacing,
        target_w=target_w,
        axial_pos=axial_pos,
        coronal_pos=coronal_pos,
        sagittal_pos=sagittal_pos,
    )
    return StreamingResponse(io.BytesIO(png), media_type="image/png")


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("MedSAM2 Web Interface starting on http://localhost:9000")
    print(f"Device: {get_device()}")
    print(f"Checkpoints dir: {CHECKPOINT_DIR}")
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=False)
