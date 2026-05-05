"""
Patient data matcher - extracts patient ID from filenames and matches with clinical data
"""
import re
import pandas as pd
from pathlib import Path

def extract_patient_id(filename):
    """
    Extract patient ID from ADNI filename
    Examples:
    - ADNI_005_S_0222_MR_... -> 005_S_0222
    - ADNI_005_S_0222_PT_... -> 005_S_0222
    - 005_S_0222_... -> 005_S_0222
    """
    # Pattern 1: ADNI_XXX_S_XXXX format (works for both MR and PT)
    match = re.search(r'ADNI[_-](\d{3}_S_\d{4})', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Pattern 2: Just XXX_S_XXXX format anywhere in filename
    match = re.search(r'(\d{3}_S_\d{4})', filename)
    if match:
        return match.group(1)
    
    # Pattern 3: Extract Image ID (I followed by digits)
    match = re.search(r'I(\d{5,6})', filename)
    if match:
        return f"IMAGE_{match.group(1)}"  # Return as IMAGE_XXXXX for lookup
    
    return None

def extract_image_id(filename):
    """
    Extract Image ID from filename
    Example: I26325_... -> 26325
    """
    match = re.search(r'I(\d{5,6})', filename)
    if match:
        return int(match.group(1))
    return None

def load_patient_clinical_data(csv_path='MMSE_data.csv'):
    """
    Load patient clinical data from MMSE CSV
    Returns: DataFrame with patient_id, mmse_score, visit_date
    """
    try:
        # Try local path first, then fallback to parent directory
        from pathlib import Path
        
        if not Path(csv_path).exists():
            csv_path = '../DataSet/MMSE_18Feb2026.csv'
        
        df = pd.read_csv(csv_path)
        
        # Get baseline (screening) visit for each patient
        baseline = df[df['VISCODE'].isin(['sc', 'bl'])].copy()
        
        # Group by patient and get most recent baseline MMSE
        baseline = baseline.sort_values('VISDATE', ascending=False)
        baseline = baseline.groupby('PTID').first().reset_index()
        
        # Select relevant columns
        patient_data = baseline[['PTID', 'MMSCORE', 'VISDATE']].copy()
        patient_data.columns = ['patient_id', 'mmse_score', 'visit_date']
        
        # Remove NaN scores
        patient_data = patient_data.dropna(subset=['mmse_score'])
        
        return patient_data
    except Exception as e:
        print(f"Error loading patient data: {e}")
        return None

def get_patient_info(patient_id, patient_data_df):
    """
    Get patient clinical information by patient ID
    Returns: dict with mmse_score, visit_date or None
    """
    if patient_data_df is None:
        return None
    
    # Match patient ID (handle different formats)
    patient_row = patient_data_df[patient_data_df['patient_id'] == patient_id]
    
    if len(patient_row) == 0:
        return None
    
    return {
        'mmse_score': int(patient_row.iloc[0]['mmse_score']),
        'visit_date': patient_row.iloc[0]['visit_date']
    }

def lookup_patient_by_image_id(image_id, mri_csv_path='../mri_full_with_image_id.csv'):
    """
    Lookup patient ID using Image ID from MRI metadata CSV
    Returns: patient_id or None
    """
    try:
        from pathlib import Path
        if not Path(mri_csv_path).exists():
            return None
            
        df = pd.read_csv(mri_csv_path)
        match = df[df['Image ID'] == image_id]
        
        if len(match) > 0:
            return match.iloc[0]['Subject ID']
        return None
    except Exception as e:
        print(f"Error looking up image ID: {e}")
        return None

def validate_patient_match(mri_filename, fdg_filename):
    """
    Validate that MRI and FDG scans are from the same patient
    Returns: (is_valid, patient_id, error_message)
    """
    mri_patient_id = extract_patient_id(mri_filename)
    fdg_patient_id = extract_patient_id(fdg_filename)
    
    # Try image ID lookup if patient ID not found
    if mri_patient_id and mri_patient_id.startswith("IMAGE_"):
        image_id = int(mri_patient_id.replace("IMAGE_", ""))
        looked_up_id = lookup_patient_by_image_id(image_id)
        if looked_up_id:
            mri_patient_id = looked_up_id
    
    if fdg_patient_id and fdg_patient_id.startswith("IMAGE_"):
        image_id = int(fdg_patient_id.replace("IMAGE_", ""))
        looked_up_id = lookup_patient_by_image_id(image_id)
        if looked_up_id:
            fdg_patient_id = looked_up_id
    
    if mri_patient_id is None:
        return False, None, f"Could not extract patient ID from MRI filename. Please rename file to include patient ID (e.g., ADNI_005_S_0222_MR_...)"
    
    if fdg_patient_id is None:
        return False, None, f"Could not extract patient ID from FDG filename. Please rename file to include patient ID (e.g., ADNI_005_S_0222_PT_...)"
    
    if mri_patient_id != fdg_patient_id:
        return False, None, f"Patient ID mismatch! MRI: {mri_patient_id}, FDG: {fdg_patient_id}. Please upload scans from the same patient."
    
    return True, mri_patient_id, None
