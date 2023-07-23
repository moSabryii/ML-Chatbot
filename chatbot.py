#importing the necessary liberaries
import json
import numpy as np
import random
import tkinter as tk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load intents from JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)['intents']

# Prepare data for the model
sentences = []
labels = []
responses = {}
for intent in intents:
    if 'intent' in intent and 'text' in intent and 'responses' in intent:
        intent_name = intent['intent']
        for pattern in intent['text']:
            sentences.append(pattern)
            labels.append(intent_name)
        responses[intent_name] = intent['responses']

# Vectorize the sentences and convert to dense array
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences).toarray()

# Transform labels to one-hot encoded format
encoder = LabelBinarizer()
y = encoder.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(vectorizer.get_feature_names()),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(encoder.classes_), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5')

# Use the model to predict the intent of a text message
def predict_intent(text):
    X_test = vectorizer.transform([text]).toarray()
    y_pred = model.predict(X_test)
    return encoder.classes_[np.argmax(y_pred)]

# Get a response for an intent
def get_response(intent_name):
    return random.choice(responses[intent_name])

# Function to insert messages to chat window
def insert_message(user, text, bg_color, fg_color):
    chatbox.insert(tk.END, user + ": " + text + '\n')
    chatbox.tag_add(user, "end-2l", "end-1l")
    chatbox.tag_config(user, background=bg_color, foreground=fg_color)

# Function to execute the chatbot and update the GUI
def chat():
    text = entry.get()
    intent_name = predict_intent(text)
    response = get_response(intent_name)
    insert_message("You", text, "#0080ff", "#ffffff")  # User messages in blue
    insert_message("Bot", response, "#2c2c2c", "#ffffff")  # Bot messages in dark grey
    entry.delete(0, tk.END)
    chatbox.yview(tk.END)  # Auto-scroll to the bottom

# Create a simple GUI
root = tk.Tk()
root.title("Chatbot")
root.geometry("400x500")
root.configure(bg="#1c1c1c")

# Create chat window
chatbox = tk.Text(root, bd=1, bg="#2c2c2c", width=50, height=8, font=("Arial", 12), foreground="#ffffff")
chatbox.place(height=385, width=370, y=6, x=6)

# Bind scrollbar to Chat window
scrollbar = tk.Scrollbar(root, command=chatbox.yview, bg="#2c2c2c")
scrollbar.place(x=375,y=6, height=385)

# Create Buttons & Entry Box
entry = tk.Entry(root, bd=0, bg="#2c2c2c",width=26, font=("Arial", 13), foreground="#ffffff")
entry.bind("<Return>", lambda event: chat())
entry.place(x=128, y=400, height=88, width=260)
button = tk.Button(root, text="Send", width=12, height=5, bd=0, bg="#0080ff", activebackground="#00bfff",foreground='#ffffff',font=("Arial", 12), command=chat)
button.place(x=6, y=400, height=88)

# Run the chatbot if the script is executed
if __name__ == "__main__":
    root.mainloop()