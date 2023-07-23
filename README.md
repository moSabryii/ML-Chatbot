# ML-Chatbot
# Chatbot Project

This project uses machine learning to create a simple chatbot that can respond to user inputs based on trained intents.

## Project Structure

The project is structured as follows:

- `chatbot.py` - The main Python script where the chatbot model is trained and the chatbot GUI is implemented.
- `intents.json` - A JSON file containing the intents, patterns, and responses for the chatbot.
- `chatbot_model.h5` - The trained model file saved in H5 format.

## Dependencies

This project uses the following libraries:

- `json`
- `numpy`
- `random`
- `tkinter`
- `tensorflow`
- `sklearn`

Please make sure to install these libraries before running the script.

## How to Run

1. Open a terminal.
2. Navigate to the project directory.
3. Run the `chatbot.py` script with the command `python chatbot.py`.
4. The chatbot GUI should appear and you can start chatting with the bot.

## How it Works

Here's a brief overview of how the chatbot works:

1. The intents, patterns, and responses are loaded from the `intents.json` file.
2. The patterns are vectorized using the TF-IDF method and the labels are one-hot encoded.
3. The data is split into a training set and a testing set.
4. A neural network model is defined and compiled.
5. The model is trained on the training data and saved to the `chatbot_model.h5` file.
6. When chatting with the bot, the text input from the user is vectorized and fed into the model to predict the intent. A random response from the predicted intent is then returned by the bot.
