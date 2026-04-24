# Naïve Bayes Spam Classifier Web App

## Overview
This project is a web application that classifies messages as spam or ham (not spam) using a Naïve Bayes classifier.

Users can:
- Enter a message and get a prediction
- View the training dataset (message + label)
- Add their own labeled messages to improve the model

---

## How It Works

This project uses a Naïve Bayes classification model.

Input: a message  
Features: words extracted from the message  
Output: spam or ham  

The model calculates:
- Prior probability P(Y)
- Likelihood P(Fi | Y)

Then predicts:
P(Y | F1, F2, ..., Fn)

The label with the higher probability is selected.

---

## Features

- Web interface using Flask
- Message classification with probability output
- Dataset visualization (table view)
- User input for adding new training data
- Model retrains dynamically when new data is added

---

## Project Structure

naive-bayes-spam-app/
│
├── app.py
├── data.csv
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── README.md

---

## Installation & Running

1. Install Flask:
pip install flask

2. Run the app:
python app.py

3. Open browser:
http://127.0.0.1:5000

---

## Example Inputs

Spam:
- free money now
- urgent claim reward

Ham:
- meeting tomorrow
- let's study later

---

## Limitations

- Assumes word independence (Naïve Bayes assumption)
- Small dataset → limited accuracy
- Retrains model from scratch when adding data
- No validation for user-submitted data

---

## Future Improvements

- Use a larger real-world dataset
- Add better feature extraction (TF-IDF)
- Improve UI (charts, highlighting)
- Deploy the app online

---

## Acknowledgment

This project was developed with assistance from generative AI tools. All code and logic were reviewed and understood before submission.