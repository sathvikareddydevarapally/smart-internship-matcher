# Smart Internship Matcher

A machine-learning-based career exploration web application that recommends a starting career track based on a student's technical skills, projects, coding practice, and CGPA.

## Features

- Career-track recommendations: Software, Data/ML, and Web Development
- Personalized skill-improvement suggestions
- Flask-based web interface
- Logistic Regression and Random Forest comparison
- Five-fold cross-validation and held-out test evaluation
- Dataset distribution and confusion matrix visualizations

## Tech Stack

Python · Flask · Pandas · NumPy · Scikit-learn · Matplotlib · HTML · CSS

## Model Evaluation

The prototype uses 1,200 synthetically generated examples. Random Forest achieved 97.9% accuracy on a held-out set of 240 synthetic examples.

Five-fold cross-validation macro-F1:
- Random Forest: 0.984
- Logistic Regression: 0.973
- Majority-class baseline: 0.286

## Limitations

The dataset and labels are synthetic and generated using illustrative scoring rules. These metrics measure agreement with those rules, not real-world internship suitability or hiring success.

The dataset is imbalanced, with Software as the majority class. Independent, realistically labelled data is needed before drawing conclusions about real-world performance.

## Run Locally

1. Install Python.
2. Create and activate a virtual environment.
3. Install dependencies:

   `python -m pip install flask pandas numpy scikit-learn matplotlib`

4. Train the model:

   `python train_model.py`

5. Start the application:

   `python app.py`

6. Open `http://127.0.0.1:5000` in your browser.

## Future Improvements

- Evaluate on independent, realistically labelled data.
- Improve class balance and assess generalization.
- Add automated tests and deployment.
- Expand career tracks and recommendation explanations.
