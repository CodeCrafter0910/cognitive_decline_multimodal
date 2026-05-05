# Smart Patient Matching System

## Overview
Automatically extracts patient IDs from uploaded MRI and FDG-PET scan filenames and matches them with clinical data (MMSE scores) from the database.

## How It Works

### 1. **File Upload**
User uploads:
- MRI scan (e.g., `ADNI_005_S_0222_MR_MPR-R__GradWarp__B1_Correction__Mask_Br_20070517132433537_S11753_I54691.nii`)
- FDG-PET scan (e.g., `ADNI_005_S_0222_FDG_PET_...nii.gz`)

### 2. **Patient ID Extraction**
System extracts patient ID from filenames using regex:
- Pattern: `XXX_S_XXXX` (e.g., `005_S_0222`)
- Works with standard ADNI filename formats

### 3. **Validation**
- Checks if both files are from the **same patient**
- Shows error if patient IDs don't match
- Example: MRI from `005_S_0222` + FDG from `006_S_0123` = ❌ ERROR

### 4. **Database Lookup**
- Searches `MMSE_data.csv` for patient ID
- Loads baseline MMSE score and visit date
- Auto-fills clinical data fields

### 5. **Display**
Shows:
- ✅ Patient ID: `005_S_0222`
- ✅ MMSE Score: `27` (Auto-loaded)
- ✅ Visit Date: `2007-05-17`

## Benefits

✅ **Zero Manual Entry** - No typing MMSE scores  
✅ **Prevents Errors** - Validates files match same patient  
✅ **Fast** - Instant lookup from database  
✅ **Professional** - Like real clinical systems  
✅ **Safe** - Shows patient info for confirmation  

## Files

- `patient_matcher.py` - Core matching logic
- `MMSE_data.csv` - Patient clinical database (2,252 patients)
- `dashboard.py` - Updated UI with smart matching

## Example Workflow

```
1. User uploads MRI: ADNI_005_S_0222_MR_...nii
2. User uploads FDG: ADNI_005_S_0222_FDG_...nii.gz
3. System extracts: Patient ID = 005_S_0222
4. System validates: ✅ Both files match
5. System loads: MMSE = 27, Visit = 2007-05-17
6. User clicks: "Predict with Full Multimodal System"
7. System predicts: CN (Cognitively Normal) - 85% confidence
```

## Fallback Behavior

If patient not found in database:
- Shows warning message
- Allows manual MMSE entry
- Continues with prediction

## Supported Filename Formats

✅ `ADNI_XXX_S_XXXX_...`  
✅ `ADNI-XXX-S-XXXX-...`  
✅ `ADNIXXX_S_XXXX...`  

❌ Image ID only (e.g., `I26325_...`) - requires manual entry
