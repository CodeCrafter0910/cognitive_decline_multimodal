# 🚀 Deployment Guide - ADNI Multimodal AI

## Quick Deploy to Streamlit Cloud (5 minutes)

### Step 1: Push to GitHub

```bash
# Initialize git (if not already)
cd adni_project
git init

# Add files
git add .
git commit -m "ADNI Multimodal AI - Ready for deployment"

# Create GitHub repo and push
git remote add origin https://github.com/YOUR_USERNAME/adni-multimodal-ai.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click "New app"
3. Connect your GitHub account
4. Select:
   - **Repository**: `YOUR_USERNAME/adni-multimodal-ai`
   - **Branch**: `main`
   - **Main file path**: `dashboard.py`
5. Click "Deploy"!

**Your app will be live at**: `https://YOUR_USERNAME-adni-multimodal-ai.streamlit.app`

---

## What Gets Deployed

```
adni_project/
├── dashboard.py           # Streamlit app
├── config.py             # Configuration
├── requirements.txt      # Dependencies
├── README.md             # Documentation
├── .streamlit/           # Streamlit config
│   └── config.toml
├── preprocessing/        # Source modules
├── models/
├── fusion/
├── evaluation/
├── utils/
└── outputs/results/      # Pre-trained models & results
```

**Size**: ~50-100 MB (manageable for deployment)

---

## Important Notes

### ✅ What Works:
- Dashboard displays all results
- Pre-trained models loaded
- All visualizations shown
- No dataset needed!

### ❌ What Doesn't Work:
- Training new models (dataset too large)
- Running `run.py` (requires local dataset)

### 💡 Solution:
- Dashboard uses **pre-generated results** from `outputs/results/`
- Models are **already trained** and saved
- Perfect for **demonstration** and **review**!

---

## Alternative Deployment Options

### Option 2: Hugging Face Spaces

1. Create account at https://huggingface.co/
2. Create new Space (Streamlit)
3. Upload files or connect GitHub
4. Auto-deploys!

### Option 3: Render

1. Create account at https://render.com/
2. New Web Service
3. Connect GitHub
4. Build command: `pip install -r requirements.txt`
5. Start command: `streamlit run dashboard.py`

### Option 4: Railway

1. Create account at https://railway.app/
2. New Project from GitHub
3. Auto-detects Streamlit
4. Deploys automatically!

---

## Troubleshooting

### Issue: "Module not found"
**Solution**: Check `requirements.txt` has all dependencies

### Issue: "File not found"
**Solution**: Ensure `outputs/results/` folder is committed to GitHub

### Issue: "Memory limit exceeded"
**Solution**: Remove large model files, keep only visualizations

---

## Live Demo URL

After deployment, your app will be accessible at:
- **Streamlit Cloud**: `https://YOUR_USERNAME-adni-multimodal-ai.streamlit.app`
- **Hugging Face**: `https://huggingface.co/spaces/YOUR_USERNAME/adni-multimodal-ai`

Share this URL with your reviewers! 🎉

---

## For Final Review

**What to show**:
1. Live dashboard URL
2. Results visualizations
3. Model performance metrics
4. Attention weight analysis
5. Cross-validation results

**What to explain**:
- Smart MRI training approach (207 patients)
- 79.3% fusion accuracy achievement
- Attention-based fusion mechanism
- Robust 5-fold cross-validation

---

**Status**: ✅ Ready to deploy!
**Estimated time**: 5-10 minutes
**Cost**: FREE (all platforms have free tiers)
