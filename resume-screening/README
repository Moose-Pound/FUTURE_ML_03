# Resume Screening and Candidate Ranking System

An AI-powered resume screening system that uses Natural Language Processing (NLP) and Machine Learning to classify resumes and rank candidates against job profiles.

## Overview

This project explores how machine learning can support recruitment by automating two tasks:

- **Resume Classification:** Predicts the occupational category of a resume using its text content.
- **Candidate Ranking:** Compares resume skills with job profile requirements and ranks the most relevant candidates for each profile.

The project combines text preprocessing, TF-IDF feature extraction, machine learning classification, and cosine similarity-based ranking.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- NLP and TF-IDF
- Matplotlib and Plotly
- Streamlit

## Machine Learning Models

Two classification models were developed and evaluated:

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| Logistic Regression | 65.52% | 60.18% |
| Linear SVC | 70.97% | 67.61% |

Linear SVC achieved the stronger classification performance in the initial model comparison.

A separate grouped evaluation achieved an accuracy of **71.37%** and a macro F1-score of **67.01%**, with no overlap between training and test groups.

## Candidate Ranking

Candidate ranking uses cosine similarity between binary skill vectors to measure the similarity between resume skills and job profile requirements.

The system generates up to five ranked job profile recommendations per resume. Skill coverage, matched skills, missing skills, and zero-similarity flags are also included as diagnostic information.

The final recommendation output contains **12,405 recommendations across 2,481 resumes**.

## Project Structure

```text
resume-screening/
├── data/
│   └── Resume.csv
├── models/
│   ├── linear_svc.pkl
│   ├── logistic_regression.pkl
│   └── tfidf_vectorizer.pkl
├── notebook/
│   └── resume_screening.system.ipynb
├── outputs/
│   ├── artifacts/
│   ├── recommendations.csv
│   └── evaluation results and visualisations
├── app.py
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Moose-Pound/FUTURE_ML_03.git
cd FUTURE_ML_03/resume-screening
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The project uses resume and job-description datasets for classification and candidate ranking.

The large `job_descriptions.csv` dataset is excluded from this GitHub repository because of its size. Place the required dataset in the `data/` directory before running the relevant notebook cells or application workflows.

## Model Artifacts and Evaluation

Trained classification models, the TF-IDF vectorizer, ranking artifacts, evaluation results, recommendation outputs, and visualisations are included in the project.

The notebook contains the data preparation, model development, evaluation, and candidate-ranking workflow.

## Future Improvements

- Evaluate ranking quality against labelled recruitment outcomes.
- Improve handling of resumes with limited or missing skill information.
- Explore additional NLP representations and ranking approaches.
- Extend evaluation across more diverse job profiles.

## Author

**Cavin Netshiavha**

GitHub: [Moose-Pound](https://github.com/Moose-Pound)

LinkedIn: [Cavin Netshiavha](https://www.linkedin.com/in/cavin-netshiavha-538ab4398/)

