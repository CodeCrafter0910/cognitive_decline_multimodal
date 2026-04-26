from pathlib import Path
import pandas as pd


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["RID"]       = df["RID"].astype(int)
    df["DIAGNOSIS"] = df["DIAGNOSIS"].str.strip()
    df["MMSCORE"]   = df["MMSCORE"].astype(float)
    return df


def rid_to_subject_folder(adni_root: Path, rid: int) -> Path | None:
    rid_padded = str(rid).zfill(4)
    for folder in adni_root.iterdir():
        if folder.is_dir() and folder.name.split("_")[-1] == rid_padded:
            return folder
    return None


def find_mri_nii(subject_folder: Path) -> Path | None:
    for f in subject_folder.rglob("*.nii"):
        if "_MR_" in f.name or "MPR" in f.name.upper() or "GRAD" in f.name.upper():
            return f
    for f in subject_folder.rglob("*.nii.gz"):
        if "_MR_" in f.name or "MPR" in f.name.upper() or "GRAD" in f.name.upper():
            return f
    all_nii = list(subject_folder.rglob("*.nii")) + list(subject_folder.rglob("*.nii.gz"))
    return all_nii[0] if all_nii else None


def find_fdg_nifti(subject_folder: Path) -> Path | None:
    """
    Find FDG PET NIfTI file (already converted).
    Looks for .nii or .nii.gz files in folders with FDG-related names.
    """
    for folder in subject_folder.iterdir():
        if not folder.is_dir():
            continue
        name = folder.name.upper()
        if "COREG" in name or "FDG" in name or "PET" in name:
            # Look for NIfTI files
            nii_files = list(folder.rglob("*.nii")) + list(folder.rglob("*.nii.gz"))
            if nii_files:
                # Return the first one (or you could filter by name)
                return nii_files[0]
    
    # Fallback: search all NIfTI files with FDG/PET/Coreg in name
    all_nii = list(subject_folder.rglob("*.nii")) + list(subject_folder.rglob("*.nii.gz"))
    for f in all_nii:
        name_upper = f.name.upper()
        if "COREG" in name_upper or "FDG" in name_upper or "PET" in name_upper:
            return f
    
    return None


def find_fdg_dicom_dir(subject_folder: Path) -> Path | None:
    """
    Find FDG PET DICOM directory (for conversion).
    This is kept for backward compatibility but may not be needed if files are already NIfTI.
    """
    for folder in subject_folder.iterdir():
        if not folder.is_dir():
            continue
        name = folder.name.upper()
        if "COREG" in name or "FDG" in name or "PET" in name:
            dcm_files = list(folder.rglob("*.dcm"))
            if dcm_files:
                return dcm_files[0].parent
    all_dcm = list(subject_folder.rglob("*.dcm"))
    return all_dcm[0].parent if all_dcm else None


def build_manifest(adni_root: Path, csv_path: Path, require_all_modalities: bool = True) -> pd.DataFrame:
    """
    Build manifest of subjects with their data paths.
    
    Args:
        adni_root: Root directory of ADNI dataset
        csv_path: Path to CSV with RID, DIAGNOSIS, MMSCORE
        require_all_modalities: If True, only return subjects with MRI+FDG+Clinical.
                                If False, return all subjects (missing modalities will be None)
    
    Returns:
        DataFrame with subject information and data paths
    """
    df   = load_csv(csv_path)
    rows = []
    
    no_folder_count = 0
    no_mri_count = 0
    no_fdg_count = 0

    for _, row in df.iterrows():
        rid        = int(row["RID"])
        diagnosis  = row["DIAGNOSIS"]
        mmscore    = float(row["MMSCORE"])
        subj_dir   = rid_to_subject_folder(adni_root, rid)

        if subj_dir is None:
            print(f"  [NO FOLDER] RID={rid}")
            no_folder_count += 1
            if require_all_modalities:
                continue
            # If not requiring all, still add with None paths
            rows.append({
                "rid":        rid,
                "subject_id": f"RID_{rid:04d}",
                "diagnosis":  diagnosis,
                "label":      {"CN": 0, "MCI": 1, "AD": 2}[diagnosis],
                "mmscore":    mmscore,
                "mri_path":   None,
                "fdg_dcm_dir": None,
            })
            continue

        mri_path = find_mri_nii(subj_dir)
        fdg_path = find_fdg_nifti(subj_dir)  # Look for NIfTI directly
        
        if mri_path is None:
            no_mri_count += 1
        if fdg_path is None:
            no_fdg_count += 1

        rows.append({
            "rid":        rid,
            "subject_id": subj_dir.name,
            "diagnosis":  diagnosis,
            "label":      {"CN": 0, "MCI": 1, "AD": 2}[diagnosis],
            "mmscore":    mmscore,
            "mri_path":   str(mri_path) if mri_path else None,
            "fdg_path":   str(fdg_path) if fdg_path else None,  # Changed from fdg_dcm_dir
        })

    manifest = pd.DataFrame(rows)
    
    # Statistics
    has_mri = manifest["mri_path"].notna().sum()
    has_fdg = manifest["fdg_path"].notna().sum()  # Changed from fdg_dcm_dir
    paired  = (manifest["mri_path"].notna() & manifest["fdg_path"].notna()).sum()

    print(f"\n  Data Coverage Report:")
    print(f"  -----------------------------------------")
    print(f"  Subjects in CSV:           {len(df)}")
    print(f"  Folders found:             {len(manifest)}")
    print(f"  Folders NOT found:         {no_folder_count}")
    print(f"  MRI available:             {has_mri}")
    print(f"  MRI missing:               {len(manifest) - has_mri}")
    print(f"  FDG available:             {has_fdg}")
    print(f"  FDG missing:               {len(manifest) - has_fdg}")
    print(f"  Clinical available (all):  {len(manifest)}")
    print(f"  Fully paired (MRI+FDG+Clin): {paired}")
    print(f"  -----------------------------------------")
    
    if require_all_modalities:
        filtered = manifest[manifest["mri_path"].notna() & manifest["fdg_path"].notna()]
        excluded = len(manifest) - len(filtered)
        if excluded > 0:
            print(f"  ⚠️  Excluding {excluded} subjects with missing MRI or FDG")
            print(f"  ✓  Using {len(filtered)} subjects with all 3 modalities")
        else:
            print(f"  ✓  All {len(filtered)} subjects have all 3 modalities!")
        return filtered.reset_index(drop=True)
    else:
        print(f"  ℹ️  Returning all {len(manifest)} subjects (some may have missing modalities)")
        return manifest
