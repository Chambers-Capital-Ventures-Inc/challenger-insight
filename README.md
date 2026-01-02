# Neuroscience-Driven Learning Engagement System:

## Challenger Insight:

Prototype predictive model for learner disengagement risk using EEG brain-wave data and learner surveys. The system applies dimensionality reduction and lightweight machine learning models to link cognitive load with engagement and provide adaptive learning triggers.

---

## Project Highlights:

- Built a machine learning pipeline to detect learner disengagement by combining EEG-derived features with self-reported survey signals.
- Achieved strong predictive performance (≈ 80%+ accuracy on held-out data) while maintaining interpretability for non-technical stakeholders.
- Identified early warning indicators of disengagement that can trigger adaptive learning interventions in real time.
- Designed the system with fairness and transparency in mind, aligning with enterprise training constraints.

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/<org-or-user>/neuroscience-driven-engagement.git
cd neuroscience-driven-engagement
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Dataset access

- EEG feature files and anonymized survey data are stored in the /data directory.
- Due to privacy constraints, raw EEG signals are not included.
- Processed feature matrices are provided for reproducibility.

### 5. Run notebooks / scripts

```bash
jupyter notebook
```

Open the notebooks in the following order:

- 01_data_exploration.ipynb
- 02_preprocessing_feature_engineering.ipynb
- 03_model_training.ipynb
- 04_evaluation_and_fairness.ipynb

---

## Project Overview

This project was developed to explore how neuroscience-informed data can improve learning outcomes in professional training environments. Our primary goal was to identify disengagement during technical learning sessions and recommend adaptive strategies before learners fully disengage. By combining EEG-based attention indicators with survey responses, we aimed to move beyond traditional completion metrics and capture engagement as it happens. This system can be contributed to reducing training drop-off, improve learner retention, and personalize instruction at scale.

---

## Data Exploration

### Datasets

- EEG-derived features: numerical indicators capturing attention, cognitive load, and signal variability.
- Learner surveys: Likert-scale and categorical responses reflecting perceived engagement, difficulty, and fatigue.

### EDA & Preprocessing

- Cleaned missing and noisy values using median imputation and normalization.
- Explored correlations between EEG features and self-reported engagement.
- Encoded categorical survey responses and aligned timestamps across modalities.

### Key EDA Insights

- Certain EEG features showed strong correlation with self-reported disengagement.
- Engagement drops often occurred before learners explicitly reported difficulty, suggesting predictive potential.

![unnamed (7)](https://github.com/user-attachments/assets/a824290f-260c-438c-afd0-11e6ff76cb9f)
![unnamed (8)](https://github.com/user-attachments/assets/f90f2fab-3b56-4a2f-b5de-f7c55df1e17d)
<img width="512" height="290" alt="unnamed (2)" src="https://github.com/user-attachments/assets/ded377ab-9714-495c-880e-0cb168e2c1d4" />

### Challenges & Assumptions

- Limited sample size required careful validation.
- EEG features were assumed to be representative proxies for attention rather than direct cognitive states.

---

## Model Development

- Tested multiple models including Logistic Regression, Random Forests, and Decision Trees.
- Selected interpretable models to support stakeholder trust and explainability.
- Applied feature selection and hyperparameter tuning using cross-validation.

### Training Setup

- 70/30 train-test split
- Primary metrics: Accuracy and F1-score
- Baseline compared against majority-class prediction

---

## Results & Key Findings

- The final model achieved ~80% accuracy with balanced precision and recall.
- EEG features related to sustained attention were among the strongest predictors.
- Fairness checks showed consistent performance across demographic subgroups, with no significant bias detected.
- These results demonstrate that neuroscience-informed features can meaningfully enhance engagement prediction when paired with traditional survey data.

---

## Dashboard Overview

The Streamlit dashboard serves as the interactive interface for the Neuroscience-Driven Learning Engagement System (Challenger Insight). It allows users to explore engagement predictions, EEG feature trends, and risk signals in a clear, interpretable format.

### Key Dashboard Features

- Engagement Risk Score: Displays predicted disengagement risk based on EEG-derived features and survey inputs.
- EEG Feature Visualizations: Interactive plots showing attention, cognitive load, and signal variability trends.
- Model Output Transparency: Simple explanations of model predictions to support non-technical stakeholders.
- Adaptive Trigger Indicators: Highlights when engagement risk crosses predefined thresholds that would prompt learning interventions.

### Dashboard

https://youtu.be/JCvjSIs3CaY

These visual components are designed to support real-time monitoring and future integration into enterprise learning platforms.

---

## Next Steps

- Expand the dataset to include longer training sessions and more participants.
- Explore temporal models (e.g., LSTMs) to capture engagement trends over time.
- Integrate the model into a live dashboard to trigger real-time adaptive learning interventions.
- Incorporate additional explainability tools to further support transparency.

---

## Team Members:

| Name                 | GitHub Handle        | Contribution                                                                                             |
| -------------------- | -------------------- | -------------------------------------------------------------------------------------------------------- |
| Meagan Alfaro        | @meaganalfaro        | Model development, prototype implementation, hyperparameter tuning, documentation                                               |
| Ishrat Arshad        | @ishratarshad        | Model development/validation, data preprocessing, exploratory data analysis (EDA), dataset documentation        |
| Ruby Hong            | @rubysmac            | Model development/validation, data preprocessing, exploratory data analysis (EDA), dataset documentation |
| Aneesa Ayornu        | @aeza404             | Data preprocessing, feature engineering, data validation                                                 |
| Mahathi Chinthapalli | @MahathiChinthapalli | Model selection, hyperparameter tuning, training and optimization                                        |
| Amanda Thomas        | @amandathom          | Model evaluation, hyperparameter tuning, performance analysis, results interpretation                                           |
| Izabella Doser       | @Izabelladesign      | Data exploration, dimensionality reduction, model development, documentation                             |
| Arian Bahram         | @ariansbahram        | Data preprocessing, exploratory data analysis (EDA), model development                                    |
| Saket Kolluru        | @Saketk2             | Dashboard development, data visualization, results interpretation                                        |
| Eden Radulescu       | @edenradulescu       | Dashboard development, data visualization, results interpretation                                        |

---
## License

This project is licensed under the MIT License (pending Challenge Advisor approval).

---

## References

- Break Through Tech AI Program materials
- Selected literature on EEG-based engagement detection and learning analytics

---

## Acknowledgements

We thank Chris Chambers (Chambers Capital Ventures) for guiding the problem scope, Lori Kirkland for introducing the real-world challenge context, and Rashidah Carr, our AI Studio Coach, for continuous feedback and support throughout the semester.
