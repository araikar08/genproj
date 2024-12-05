# RNN for Last Name Classification

## Project Overview

This project is about training a neural network to predict the language of origin for a given last name written in Latin alphabets. We wanted to see if a model could learn patterns from surnames and figure out what language they are most likely from. We use a Recurrent Neural Network (RNN) because last names can be treated as sequences of characters, which RNNs are good at handling. We also added an option to use an LSTM model, which is an advanced version of an RNN, but for now, we mainly trained the basic RNN version.

## Main Goals

- Preprocess last name data to make it usable by our neural network.
- Use PyTorch to implement and train an RNN (with an optional LSTM implementation too).
- Evaluate the model to see how well it can predict the language of origin for new, unseen last names.

## Dataset

We used a dataset of last names organized by language. The data is downloaded directly from PyTorch's tutorial resources. The dataset contains text files where each file lists last names for a specific language. These include languages like English, French, Chinese, and many more. We convert the names from Unicode to ASCII, then encode each character as an integer so the model can work with it.

## Model Architecture

We built the model using PyTorch, and it includes the following parts:

1. **Embedding Layer**: Converts each character to a vector of real numbers, so the model has a better representation to work with.
2. **RNN/LSTM Layer**: Processes the sequence of character embeddings and keeps track of context as it moves through each character.
3. **Dropout Layer**: Helps prevent overfitting by randomly ignoring some units during training.
4. **Fully Connected Output Layer**: Makes the final prediction about which language category the name belongs to.

The model can be either an RNN or an LSTM, depending on the configuration, but for this project, we primarily used the basic RNN.

## Training the Model

We trained the model using the following steps:

1. **Data Splitting**: The dataset was divided into training and testing sets (80% training and 20% testing).
2. **Batch Creation**: We wrote a function to create batches for training, so the model could be trained more efficiently on multiple names at a time.
3. **Training Process**: The model was trained for 20 epochs, and we used cross-entropy loss to measure how well the model's predictions matched the actual languages. The optimizer used was Adam, which is known for being efficient for this type of task.

## Results

The model's performance was tracked over the training process, and the accuracy was evaluated after each epoch. We aimed for an evaluation accuracy of over 80%. The training was done on a GPU to speed things up since RNNs can take a while to train on sequences.

## How to Use This Project

- Clone this repository and make sure you have the required libraries installed, mainly PyTorch.
- You can train the model using the provided training script. Make sure to enable GPU for faster training.
- After training, you can test the model by inputting any last name, and it will predict the probable language of origin.

## Future Improvements

- **Use LSTM**: Try switching the model to LSTM to see if the accuracy improves.
- **More Languages**: Add more language files to see if the model can learn to classify even more categories.
- **Hyperparameter Tuning**: Experiment with different hyperparameters like learning rate, batch size, and number of layers to see if you can improve the model's performance.

## Conclusion

This project is a simple but effective way to explore character-level natural language processing using RNNs. It was interesting to see how the model learned patterns from different last names to predict their language of origin. There is definitely room for improvement, but it's a great start for someone interested in using neural networks for language classification tasks.
