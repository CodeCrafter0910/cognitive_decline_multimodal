# File Naming Guide for Predictions

## Problem
Your FDG-PET files don't have patient IDs in the filename:
- ❌ `I26325_Coreg._Avg,Standardized_Image_and_Voxel_Size_20061003123403-196150546.nii.gz`
- ✅ `ADNI_005_S_0222_PT_Coreg,_Avg,_Standardized_Image_and_Voxel_Size_Br_20061016120735362_1_S19643_I26325.dcm`

## Solution Options

### Option 1: Rename Files (Recommended)
Rename your FDG files to include the patient ID from the folder path:

**Before:**
```
DataSet(Final)/ADNI/005_S_0222/Coreg__Avg__Standardized.../I26325/I26325_Coreg...nii.gz
```

**After:**
```
ADNI_005_S_0222_PT_I26325_Coreg...nii.gz
```

**Quick Rename Script (Python):**
```python
import os
from pathlib import Path

# Navigate to your DataSet(Final)/ADNI folder
base_path = Path("C:/Users/Rishabh Khanna/OneDrive/Desktop/DataSet(Final)/ADNI")

for patient_folder in base_path.iterdir():
    if patient_folder.is_dir():
        patient_id = patient_folder.name  # e.g., "005_S_0222"
        
        # Find all .nii or .nii.gz files
        for scan_file in patient_folder.rglob("*.nii*"):
            if patient_id not in scan_file.name:
                # Rename to include patient ID
                new_name = f"ADNI_{patient_id}_{scan_file.name}"
                new_path = scan_file.parent / new_name
                scan_file.rename(new_path)
                print(f"Renamed: {scan_file.name} -> {new_name}")
```

### Option 2: Use Manual Entry
If you can't rename files:

1. Upload your MRI and FDG files
2. Check the box: **"Cannot extract patient ID? Use manual entry"**
3. Enter:
   - Patient ID: `005_S_0222`
   - MMSE Score: (will auto-load if patient in database)
   - Age: (optional)

### Option 3: Create a Mapping File
Create `image_id_mapping.csv`:
```csv
image_id,patient_id
26325,005_S_0222
54691,005_S_0222
...
```

Then the system can look up patient ID by image ID.

## Recommended Filename Format

### MRI Files:
```
ADNI_XXX_S_XXXX_MR_[description]_IXXXXX.nii
Example: ADNI_005_S_0222_MR_MPR-R__GradWarp__B1_Correction__Mask_Br_20070517132433537_S11753_I54691.nii
```

### FDG-PET Files:
```
ADNI_XXX_S_XXXX_PT_[description]_IXXXXX.nii.gz
Example: ADNI_005_S_0222_PT_Coreg,_Avg,_Standardized_Image_and_Voxel_Size_Br_20061016120735362_1_S19643_I26325.nii.gz
```

## Current System Capabilities

✅ **Extracts patient ID from:**
- `ADNI_005_S_0222_MR_...`
- `ADNI_005_S_0222_PT_...`
- `005_S_0222_...`
- Any filename with `XXX_S_XXXX` pattern

✅ **Validates:**
- Both files are from same patient
- Patient exists in MMSE database

✅ **Auto-loads:**
- MMSE score
- Visit date

❌ **Cannot extract from:**
- `I26325_...` (image ID only)
- Files without patient ID in name

## Quick Fix for Your Current Files

Run this in your DataSet(Final) folder:
```bash
# Windows PowerShell
Get-ChildItem -Recurse -Filter "*.nii*" | ForEach-Object {
    $patientId = $_.Directory.Parent.Parent.Name
    if ($patientId -match '\d{3}_S_\d{4}' -and $_.Name -notmatch $patientId) {
        $newName = "ADNI_" + $patientId + "_" + $_.Name
        Rename-Item $_.FullName -NewName $newName
        Write-Host "Renamed: $($_.Name) -> $newName"
    }
}
```
