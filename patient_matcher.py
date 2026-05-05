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
    - I26325_Coreg._Avg,Standardized_Image_and_Voxel_Size_20061003123403-196150 -> Extract from metadata
    """
    # Pattern 1: ADNI_XXX_S_XXXX format
    match = re.search(r'(\d{3}_S_\d{4})', filename)
    if match:
        return match.group(1)
    
    # Pattern 2: Try to extract from other formats
    match = re.search(r'ADNI[_-]?(\d{3}[_-]S[_-]\d{4})', filename, re.IGNORECASE)
    if match:
        return match.group(1).replace('-', '_')
    
    return None

def load_patient_clinical_data(csv_path='../DataSet/MMSE_18Feb2026.csv'):
    """
    Load patient clinical data from MMSE CSV
    Returns: DataFrame with patient_id, mmse_score, visit_date
    """
    try:
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

def validate_patient_match(mri_filename, fdg_filename):
    """
    Validate that MRI and FDG scans are from the same patient
    Returns: (is_valid, patient_id, error_message)
    """
    mri_patient_id = extract_patient_id(mri_filename)
    fdg_patient_id = extract_patient_id(fdg_filename)
    
    if mri_patient_id is None:
        return False, None, f"Could not extract patient ID from MRI filename: {mri_filename}"
    
    if fdg_patient_id is None:
        return False, None, f"Could not extract patient ID from FDG filename: {fdg_filename}"
    
    if mri_patient_id != fdg_patient_id:
        return False, None, f"Patient ID mismatch! MRI: {mri_patient_id}, FDG: {fdg_patient_id}"
    
    return True, mri_patient_id, None
