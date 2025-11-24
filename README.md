

# Chatbot Project

This project implements a simple yet effective chatbot using a machine learning-based intent classification model. The chatbot uses a JSON-based dataset containing intents, patterns, and responses and leverages a Naive Bayes classifier to predict user intents.

---

# Table of Contents
1. [Project Overview](#project-overview)
2. [Data](#data)
3. [Features](#features)
4. [Steps to Build the Chatbot](#steps-to-build-the-chatbot)
   - Import Necessary Libraries
   - Load the JSON Data
   - Extract Patterns, Tags, and Responses
   - Data Preprocessing
   - Split Data into Training and Testing Sets
   - Train the Model
   - Evaluate the Model
   - Build the Chatbot Framework
   - Run the Chatbot
5. [Installation and Setup](#installation-and-setup)
6. [Usage](#usage)
7. [Tools and Libraries](#tools-and-libraries)
8. [Contributors](#contributors)

---
#  Project Overview

This project demonstrates how to build an interactive AI-powered chatbot using machine learning techniques.  
It uses **Naive Bayes classification**, **CountVectorizer**, and a JSON-based intent dataset to classify user inputs and generate meaningful responses.  
The architecture is modular and easy to extend into LLM, RAG, or Agentic AI systems.

---

#  Data

The chatbot uses a JSON dataset (`intents.json`) that contains:

- **Tags** (intent name)
- **Patterns** (example user messages)
- **Responses** (bot replies)

Example:

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey"],
      "responses": ["Hello!", "Hi there!", "How can I help you today?"]
    }
  ]
}
```
# Features

- Intent classification using Machine Learning

- JSON-driven conversational design

- Modular ML pipeline (train → predict → respond)

- CLI-based chatbot interface

- Easy to update and extend

- Supports multiple intents and custom datasets

# Steps to Build the Chatbot

## 1. Import Necessary Libraries
```python
import json
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
```

## 2. Load the JSON Data
Load the chatbot intents from a JSON file:
```python
with open("data/intents.json", "r") as f:
    data = json.load(f)
 ```
## 3. Extract Patterns, Tags, and Responses
Process the JSON data to extract:
- Patterns: Example queries.
- Tags: Intent labels.
- Responses: Predefined replies for each intent.
```python
patterns = []
tags = []
responses = {}

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
    responses[intent['tag']] = intent['responses']
```

## 4. Data Preprocessing
Convert text patterns into numerical vectors and encode tags as numerical labels:
```python
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)  # Vectorize patterns
unique_tags = list(set(tags))
y_encoded = [unique_tags.index(tag) for tag in tags]
```

## 5. Split Data into Training and Testing Sets
Split the dataset into training and testing subsets:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
```

## 6. Train the Model
Train a Naive Bayes classifier:
```python
model = MultinomialNB()
model.fit(X_train, y_train)
```

## 7. Evaluate the Model
Evaluate the model on the test set using accuracy as a metric:
```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```

## 8. Build the Chatbot Framework
Implement a function to process user inputs, predict intents, and provide responses:
```python
def chatbot_response(user_input):
    user_input_vectorized = vectorizer.transform([user_input])
    predicted_tag_index = model.predict(user_input_vectorized)[0]
    predicted_tag = unique_tags[predicted_tag_index]
    return random.choice(responses[predicted_tag])
```

## 9. Run the Chatbot
Launch the chatbot and interact via the command line:
```python
print("Chatbot: Hello! I am your assistant. Type 'exit' to end the conversation.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye! Have a nice day!")
        break
    response = chatbot_response(user_input)
    print(f"Chatbot: {response}")
```

---

# Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chatbot-project.git
   cd chatbot-project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

# Usage

1. Run the chatbot script:
   ```bash
   python chatbot_app.py
   ```

2. Type queries to interact with the chatbot. Type `exit` to quit.

---

# Tools and Libraries

- **Python 3.7+**: Programming language for development.
- **Scikit-learn**: For vectorization and intent classification.
- **JSON**: To structure the dataset.
- **Random**: For dynamic response selection.

---
# Contributors

Maheswar Reddy Thatiparthi
AI/ML Engineer • Generative AI • Python Developer
📧 mahesHwarreddy140@gmail.com

🔗 LinkedIn: https://linkedin.com/in/maheswarreddythatiparthi


---

