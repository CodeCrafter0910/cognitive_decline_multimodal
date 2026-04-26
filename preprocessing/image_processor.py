"""
Enhanced Image Processor with Data Augmentation (Problem 14)

Upgrades:
1. Medical-imaging-specific augmentation pipeline
2. Robust preprocessing with better error handling
3. On-the-fly augmentation for training data multiplication
4. Support for both MRI and FDG-PET volumes
"""

import subprocess
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, rotate, shift, gaussian_filter


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA AUGMENTATION (Problem 14)
# ═══════════════════════════════════════════════════════════════════════════════

class MedicalAugmentation3D:
    """
    Medical imaging augmentation pipeline for 3D brain volumes.
    
    Augmentations applied:
    - Random rotation (±10°, all axes)
    - Random scaling (0.9-1.1x)
    - Random translation (±5 voxels)
    - Left-right flipping (brain symmetry)
    - Gaussian noise injection
    - Gamma correction (contrast variation)
    - Gaussian blurring (slight)
    - Elastic deformation (simulates anatomical variation)
    """
    
    def __init__(self, rotation_range=10, scale_range=(0.9, 1.1),
                 translation=5, noise_std=0.05, flip_prob=0.5,
                 gamma_range=(-0.3, 0.3), elastic_enabled=True,
                 elastic_alpha=7.5):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation = translation
        self.noise_std = noise_std
        self.flip_prob = flip_prob
        self.gamma_range = gamma_range
        self.elastic_enabled = elastic_enabled
        self.elastic_alpha = elastic_alpha
    
    def __call__(self, volume: np.ndarray, seed: int = None) -> np.ndarray:
        """
        Apply random augmentation to a 3D volume.
        
        Args:
            volume: (D, H, W) numpy array
            seed: Random seed for reproducibility (optional)
        
        Returns:
            Augmented volume (D, H, W)
        """
        if seed is not None:
            np.random.seed(seed)
        
        aug = volume.copy()
        
        # 1. Random rotation (75% probability)
        if np.random.random() < 0.75:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            axis = np.random.choice([0, 1, 2])
            axes = [(1, 2), (0, 2), (0, 1)][axis]
            aug = rotate(aug, angle, axes=axes, reshape=False, order=1, mode='constant')
        
        # 2. Random scaling (50% probability)
        if np.random.random() < 0.5:
            scale = np.random.uniform(*self.scale_range)
            scaled = zoom(aug, scale, order=1)
            # Crop or pad to original shape
            aug = self._match_shape(scaled, volume.shape)
        
        # 3. Random translation (50% probability)
        if np.random.random() < 0.5:
            shifts = np.random.uniform(-self.translation, self.translation, size=3)
            aug = shift(aug, shifts, order=1, mode='constant')
        
        # 4. Left-right flip (50% probability, brain is roughly symmetric)
        if np.random.random() < self.flip_prob:
            aug = np.flip(aug, axis=2).copy()  # Flip along LR axis
        
        # 5. Gaussian noise (25% probability)
        if np.random.random() < 0.25:
            noise = np.random.normal(0, self.noise_std, aug.shape)
            aug = aug + noise
        
        # 6. Gamma correction (50% probability)
        if np.random.random() < 0.5:
            gamma = np.exp(np.random.uniform(*self.gamma_range))
            # Shift to positive range for gamma correction
            aug_min = aug.min()
            aug_shifted = aug - aug_min + 1e-8
            aug_shifted = np.power(aug_shifted, gamma)
            aug = aug_shifted + aug_min
        
        # 7. Gaussian blur (25% probability)
        if np.random.random() < 0.25:
            sigma = np.random.uniform(0.5, 1.5)
            aug = gaussian_filter(aug, sigma=sigma)
        
        # 8. Elastic deformation (30% probability, if enabled)
        if self.elastic_enabled and np.random.random() < 0.3:
            aug = self._elastic_deformation(aug, alpha=self.elastic_alpha)
        
        return aug.astype(np.float32)
    
    def _match_shape(self, volume: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Center crop or pad volume to match target shape."""
        result = np.zeros(target_shape, dtype=volume.dtype)
        
        # Calculate crop/pad offsets
        for axis in range(3):
            if volume.shape[axis] > target_shape[axis]:
                start = (volume.shape[axis] - target_shape[axis]) // 2
                slices_src = [slice(None)] * 3
                slices_src[axis] = slice(start, start + target_shape[axis])
                volume = volume[tuple(slices_src)]
            elif volume.shape[axis] < target_shape[axis]:
                pad_before = (target_shape[axis] - volume.shape[axis]) // 2
                pad_after = target_shape[axis] - volume.shape[axis] - pad_before
                pad_width = [(0, 0)] * 3
                pad_width[axis] = (pad_before, pad_after)
                volume = np.pad(volume, pad_width, mode='constant')
        
        # Ensure exact shape match
        slices = tuple(slice(0, s) for s in target_shape)
        result[:] = volume[slices]
        return result
    
    def _elastic_deformation(self, volume: np.ndarray, alpha: float = 7.5) -> np.ndarray:
        """Simple elastic deformation for 3D volumes."""
        shape = volume.shape
        
        # Generate random displacement fields
        sigma = alpha * 0.3  # Smoothing
        dx = gaussian_filter(np.random.randn(*shape) * alpha, sigma=sigma)
        dy = gaussian_filter(np.random.randn(*shape) * alpha, sigma=sigma)
        dz = gaussian_filter(np.random.randn(*shape) * alpha, sigma=sigma)
        
        # Create coordinate grids
        x, y, z = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing='ij'
        )
        
        # Apply displacement
        coords = np.array([
            np.clip(x + dx, 0, shape[0] - 1),
            np.clip(y + dy, 0, shape[1] - 1),
            np.clip(z + dz, 0, shape[2] - 1)
        ]).astype(int)
        
        return volume[coords[0], coords[1], coords[2]]


def augment_dataset(npy_dir: Path, subject_ids: list, labels: np.ndarray,
                    n_augments: int = 3, seed: int = 42) -> tuple:
    """
    Augment training data by creating multiple transformed copies.
    
    Args:
        npy_dir: Directory containing .npy volume files
        subject_ids: List of subject IDs to augment
        labels: Labels for each subject
        n_augments: Number of augmented copies per original
        seed: Random seed
    
    Returns:
        aug_ids: Extended subject ID list
        aug_labels: Extended label array
        aug_volumes: List of augmented volume paths (saved to disk)
    """
    augmenter = MedicalAugmentation3D()
    np.random.seed(seed)
    
    aug_dir = npy_dir.parent / f"{npy_dir.name}_augmented"
    aug_dir.mkdir(parents=True, exist_ok=True)
    
    aug_ids = list(subject_ids)
    aug_labels = list(labels)
    
    total_created = 0
    
    for idx, (sid, label) in enumerate(zip(subject_ids, labels)):
        path = npy_dir / f"{sid}.npy"
        if not path.exists():
            continue
        
        vol = np.load(str(path)).astype(np.float32)
        
        for aug_idx in range(n_augments):
            aug_sid = f"{sid}_aug{aug_idx}"
            aug_path = aug_dir / f"{aug_sid}.npy"
            
            if not aug_path.exists():
                aug_vol = augmenter(vol, seed=seed + idx * n_augments + aug_idx)
                np.save(str(aug_path), aug_vol)
            
            aug_ids.append(aug_sid)
            aug_labels.append(label)
            total_created += 1
    
    print(f"  Augmentation: {len(subject_ids)} → {len(aug_ids)} samples "
          f"({total_created} augmented copies)")
    
    return aug_ids, np.array(aug_labels), aug_dir


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE PREPROCESSING (preserved from original with improvements)
# ═══════════════════════════════════════════════════════════════════════════════

def convert_dicom_to_nifti(dcm_dir: Path, out_dir: Path, subject_id: str) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["dcm2niix", "-z", "n", "-f", subject_id, "-o", str(out_dir), str(dcm_dir)],
        capture_output=True
    )
    matches = list(out_dir.glob(f"{subject_id}*.nii")) + \
              list(out_dir.glob(f"{subject_id}*.nii.gz")) + \
              list(out_dir.glob("*.nii")) + \
              list(out_dir.glob("*.nii.gz"))
    return matches[0] if matches else None


def load_nifti(path: Path) -> np.ndarray:
    """Load NIfTI file with error handling."""
    try:
        img  = nib.load(str(path))
        data = img.get_fdata(dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]
        
        # Validate data
        if data.size == 0:
            raise ValueError("Empty volume")
        if np.isnan(data).all():
            raise ValueError("All NaN values")
        
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load NIfTI from {path}: {e}")


def preprocess_volume(volume: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Preprocess volume with robust error handling."""
    try:
        # Handle NaN values
        if np.isnan(volume).any():
            volume = np.nan_to_num(volume, nan=0.0)
        
        # Clip outliers
        lo     = np.percentile(volume, 0.5)
        hi     = np.percentile(volume, 99.5)
        volume = np.clip(volume, lo, hi)

        # Normalize
        mean   = volume.mean()
        std    = volume.std() if volume.std() > 1e-8 else 1.0
        volume = (volume - mean) / std

        # Resize
        if volume.shape[:3] != target_shape:
            factors = [t / s for t, s in zip(target_shape, volume.shape[:3])]
            volume  = zoom(volume, factors, order=1)

        return volume.astype(np.float32)
    
    except Exception as e:
        raise RuntimeError(f"Preprocessing failed: {e}")


def process_mri(mri_path: Path, out_dir: Path, subject_id: str,
                target_shape: tuple, overwrite: bool = False) -> bool:
    out_path = out_dir / f"{subject_id}.npy"
    if out_path.exists() and not overwrite:
        return True
    try:
        vol = load_nifti(mri_path)
        vol = preprocess_volume(vol, target_shape)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), vol)
        return True
    except Exception as e:
        print(f"    MRI ERROR {subject_id}: {e}")
        return False


def process_fdg_nifti(nifti_path: Path, npy_dir: Path, subject_id: str, 
                      target_shape: tuple, overwrite: bool = False) -> bool:
    """
    Process FDG PET file that is already in NIfTI format.
    No DICOM conversion needed.
    """
    out_path = npy_dir / f"{subject_id}.npy"
    if out_path.exists() and not overwrite:
        return True
    try:
        vol = load_nifti(nifti_path)
        vol = preprocess_volume(vol, target_shape)
        npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), vol)
        return True
    except Exception as e:
        print(f"    FDG ERROR {subject_id}: {e}")
        return False


def process_fdg(dcm_dir: Path, nii_dir: Path, npy_dir: Path,
                subject_id: str, target_shape: tuple, overwrite: bool = False) -> bool:
    """
    Process FDG from DICOM (requires dcm2niix).
    This is kept for backward compatibility but may not be needed.
    """
    out_path = npy_dir / f"{subject_id}.npy"
    if out_path.exists() and not overwrite:
        return True
    try:
        nii_path = convert_dicom_to_nifti(dcm_dir, nii_dir / subject_id, subject_id)
        if nii_path is None:
            print(f"    FDG CONVERT FAILED {subject_id}")
            return False
        vol = load_nifti(nii_path)
        vol = preprocess_volume(vol, target_shape)
        npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), vol)
        return True
    except Exception as e:
        print(f"    FDG ERROR {subject_id}: {e}")
        return False


def run_all(manifest, mri_npy_dir: Path, fdg_npy_dir: Path, target_shape: tuple, overwrite: bool = False):
    """
    Preprocess MRI and FDG volumes.
    Now handles FDG files that are already in NIfTI format.
    """
    total  = len(manifest)
    mri_ok = 0
    fdg_ok = 0

    print(f"\nPreprocessing {total} subjects  →  target shape {target_shape}")

    for i, row in manifest.iterrows():
        sid = row["subject_id"]

        # Process MRI
        mri_done = process_mri(
            Path(row["mri_path"]), mri_npy_dir, sid, target_shape, overwrite)

        # Process FDG (now expects NIfTI path directly)
        if "fdg_path" in row and row["fdg_path"] is not None:
            fdg_done = process_fdg_nifti(
                Path(row["fdg_path"]), fdg_npy_dir, sid, target_shape, overwrite)
        else:
            fdg_done = False

        status = f"MRI={'OK' if mri_done else 'FAIL'}  FDG={'OK' if fdg_done else 'FAIL'}"
        print(f"  [{i+1:03d}/{total}]  {sid}  {status}")

        if mri_done: mri_ok += 1
        if fdg_done: fdg_ok += 1

    print(f"\nDone  —  MRI: {mri_ok}/{total}   FDG: {fdg_ok}/{total}")
