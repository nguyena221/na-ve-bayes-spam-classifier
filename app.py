from flask import Flask, render_template, request
import math
import re
from collections import defaultdict, Counter

app = Flask(__name__)

training_data = [
    ("free money now", "spam"),
    ("win a free prize", "spam"),
    ("claim your free gift", "spam"),
    ("limited time offer", "spam"),
    ("click here to win cash", "spam"),
    ("cheap loans available now", "spam"),
    ("meeting tomorrow at noon", "ham"),
    ("can you send me the notes", "ham"),
    ("let's study for the exam", "ham"),
    ("are you free for lunch", "ham"),
    ("project update attached", "ham"),
    ("see you in class tomorrow", "ham"),
]

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
            # Prior: P(label)
            log_prob = math.log(self.label_counts[label] / total_messages)

            for word in words:
                # Laplace smoothing so unknown/rare words don't break the math
                word_count = self.word_counts[label][word] + 1
                total_count = self.total_words[label] + vocab_size
                log_prob += math.log(word_count / total_count)

            scores[label] = log_prob

        # Convert log scores into probabilities
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


classifier = NaiveBayesClassifier()
classifier.train(training_data)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probabilities = None
    message = ""

    if request.method == "POST":
        message = request.form["message"]
        prediction, probabilities = classifier.predict(message)

    return render_template(
        "index.html",
        prediction=prediction,
        probabilities=probabilities,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)