from flask import Flask, render_template, request
import math
import re
import csv
import random
from collections import defaultdict, Counter

app = Flask(__name__)

def load_data(filename):
    data = []
    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append((row["message"], row["label"]))
    return data

def save_data(filename, data):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["message", "label"])
        for msg, label in data:
            writer.writerow([msg, label])

def augment_data(data):
    augmented = []
    for msg, label in data:
        words = msg.split()

        if len(words) > 2:
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]

        new_msg = " ".join(words)
        augmented.append((new_msg, label))

    return data + augmented

def tokenize(text):
    text = text.lower()
    return re.findall(r"[a-z]+", text)

class NaiveBayesClassifier:
    def __init__(self):
        self.label_counts = Counter()
        self.word_counts = defaultdict(Counter)
        self.total_words = Counter()
        self.vocabulary = set()

    def train(self, data):
        for message, label in data:
            self.label_counts[label] += 1
            words = tokenize(message)

            for word in words:
                self.word_counts[label][word] += 1
                self.total_words[label] += 1
                self.vocabulary.add(word)

    def predict(self, message):
        words = tokenize(message)
        total_messages = sum(self.label_counts.values())
        vocab_size = len(self.vocabulary)

        scores = {}

        for label in self.label_counts:
            # Prior
            log_prob = math.log(self.label_counts[label] / total_messages)

            for word in words:
                # Laplace smoothing
                word_count = self.word_counts[label][word] + 1
                total_count = self.total_words[label] + vocab_size
                log_prob += math.log(word_count / total_count)

            scores[label] = log_prob

        # Convert to probabilities
        max_score = max(scores.values())
        exp_scores = {
            label: math.exp(score - max_score)
            for label, score in scores.items()
        }

        total = sum(exp_scores.values())
        probabilities = {
            label: round(exp_scores[label] / total * 100, 2)
            for label in exp_scores
        }

        prediction = max(probabilities, key=probabilities.get)

        return prediction, probabilities

training_data = load_data("data.csv")

training_data = augment_data(training_data)

print("Training size:", len(training_data))  # sanity check

classifier = NaiveBayesClassifier()
classifier.train(training_data)

@app.route("/", methods=["GET", "POST"])
def home():
    global classifier, training_data

    prediction = None
    probabilities = None
    message = ""

    if request.method == "POST":
        if "classify" in request.form:
            message = request.form["message"]
            prediction, probabilities = classifier.predict(message)

        elif "add_data" in request.form:
            new_msg = request.form["new_message"]
            new_label = request.form["new_label"]

            if new_msg and new_label in ["spam", "ham"]:
                training_data.append((new_msg, new_label))

                # save to CSV
                save_data("data.csv", training_data)

                # retrain model
                classifier = NaiveBayesClassifier()
                classifier.train(training_data)

    return render_template(
        "index.html",
        prediction=prediction,
        probabilities=probabilities,
        message=message,
        data=training_data[:20]  # show first 20 rows
    )

if __name__ == "__main__":
    app.run(debug=True)