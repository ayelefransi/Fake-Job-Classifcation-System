#  AI Job Authenticity Scanner

<img width="1883" height="833" alt="image" src="https://github.com/user-attachments/assets/5035d71d-1d03-4343-a298-05e18e6028e4" />


A full-stack, AI-powered system designed to detect fraudulent or scam job postings. Built with **React** (Vite), **FastAPI**, and a powerful **Machine Learning Ensemble Meta-Learner** trained on thousands of real and fake job listings.

Developed by **Fransi Ayele**.

##  Features
- **Meta-Learner Architecture**: Utilizes a Stacking Classifier (XGBoost, LightGBM, Random Forest) nested inside a Logistic Regression final estimator.
- **Deep Semantic Analysis**: Uses `all-MiniLM-L6-v2` BERT embeddings to extract 384-dimensional dense vectors representing the semantic meaning of the job description.
- **Linguistic Pattern Matching**: Analyzes structural cues such as text length, requirement structure, and the presence of specific scam-related keywords (e.g. *western union, wire transfer*).
- **Stunning UI/UX**: Features a custom dark-mode glassmorphism interface built directly in React without heavy UI libraries.

##  Architecture

- `/frontend`: The **React + Vite** single-page application.
- `/backend`: The **FastAPI** Python server that handles the `/predict` inference API.
- `/Models`: Contains the serialized 26MB `fake_job_ensemble.joblib` artifact.
- `notebook_code.py`: The original Data Science and Model Training pipeline.

---

##  Local Development

### Prerequisites
- Node.js (v18+)
- Python (3.9+)

### 1. Start the FastAPI Backend
```bash
# Navigate to the project root
cd "Fake News Classifcation System"

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install exact ML dependencies
pip install -r backend/requirements.txt

# Run the server
uvicorn backend.main:app --reload --port 8000
```
The backend API will be running at `http://localhost:8000`.

### 2. Start the React Frontend
Open a new terminal window:
```bash
cd "Fake News Classifcation System/frontend"

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```
The application will be accessible at `http://localhost:5173`.

---

##  Production Deployment

### Frontend (Vercel)
The React application is heavily optimized and can be easily deployed to [Vercel](https://vercel.com/):
1. Connect this repository to Vercel.
2. In the Vercel Project Settings, set **Root Directory** to `frontend`.
3. Add an Environment Variable `VITE_API_URL` pointing to your deployed Render backend (e.g., `https://your-backend.onrender.com`).
4. Re-deploy.

### Backend (Render / Railway / Heroku)
Due to the memory requirements of loading `joblib` artifacts and running `SentenceTransformers` locally, **Vercel Serverless Functions will not work**. We heavily recommend deploying the backend as a Web Service on [Render](https://render.com/):
1. Connect the repository to Render as a **Web Service**.
2. **Root Directory**: `.` (leave blank or use repository root).
3. **Build Command**: `pip install -r backend/requirements.txt`
4. **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. The deployment will install `xgboost`, `lightgbm`, and download the `all-MiniLM-L6-v2` embeddings on boot.

---

##  The Machine Learning Pipeline
The model was trained on an aggregated dataset of over 20,000 job postings (combining datasets from various sources including Kaggle).

1. **Feature Engineering**: NLP cleanup, missing data imputation.
2. **Dense Vectorization**: Contextual sentence embedding using HuggingFace's `SentenceTransformer`.
3. **Hyperparameter Optimization**: `Optuna` was used to maximize the `AUCPR` (Area Under the Precision-Recall Curve) to handle extreme class imbalances.
4. **Ensemble Stacking**: Base models learn distinct sub-patterns, and the meta-learner optimally weights their predictions.

---
*Created as part of a Fake News / Fraud Classification research system.*
