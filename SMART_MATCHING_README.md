# Smart Patient Matching System

## Overview
Automatically extracts patient IDs from uploaded MRI and FDG-PET scan filenames and matches them with clinical data (MMSE scores) from the database.

**NEW: OR Logic** - If patient ID is found in **either** file, it will be used for both! No need for both files to have the ID.

## How It Works

### 1. **File Upload**
User uploads:
- MRI scan (e.g., `ADNI_005_S_0222_MR_MPR-R__GradWarp__B1_Correction__Mask_Br_20070517132433537_S11753_I54691.nii`)
- FDG-PET scan (e.g., `I26325_Coreg._Avg,Standardized_Image_and_Voxel_Size_...nii.gz`)

### 2. **Patient ID Extraction (OR Logic)**
System extracts patient ID from filenames using regex:
- Pattern: `XXX_S_XXXX` (e.g., `005_S_0222`)
- **NEW:** If only ONE file has patient ID, uses that for BOTH files
- Works with standard ADNI filename formats

### 3. **Validation**
- ✅ **MRI has ID, FDG doesn't** → Uses MRI's patient ID
- ✅ **FDG has ID, MRI doesn't** → Uses FDG's patient ID  
- ✅ **Both have same ID** → Perfect match
- ❌ **Both have different IDs** → Error (safety check)
- ❌ **Neither has ID** → Manual entry required

### 4. **Database Lookup**
- Searches `MMSE_data.csv` for patient ID
- Loads baseline MMSE score and visit date
- Auto-fills clinical data fields

### 5. **Display**
Shows:
- ✅ Patient ID: `005_S_0222`
- ℹ️ "Using patient ID from MRI: 005_S_0222" (if only MRI had ID)
- ✅ MMSE Score: `27` (Auto-loaded)
- ✅ Visit Date: `2007-05-17`

## Example Workflow

```
1. User uploads MRI: ADNI_005_S_0222_MR_...nii (HAS patient ID)
2. User uploads FDG: I26325_Coreg...nii.gz (NO patient ID)
3. System extracts: MRI = 005_S_0222, FDG = None
4. System uses OR logic: ✅ Uses 005_S_0222 for both
5. System loads: MMSE = 27, Visit = 2007-05-17
6. User clicks: "Predict with Full Multimodal System"
7. System predicts: CN (Cognitively Normal) - 85% confidence
```

## Benefits

✅ **Only ONE file needs patient ID** - OR logic!  
✅ **Zero Manual Entry** - No typing MMSE scores  
✅ **Prevents Errors** - Validates files match same patient  
✅ **Fast** - Instant lookup from database  
✅ **Professional** - Like real clinical systems  
✅ **Safe** - Shows patient info for confirmation  
✅ **Flexible** - Works even if one filename is incomplete

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
