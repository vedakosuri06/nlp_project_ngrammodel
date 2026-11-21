# N-Gram Next-Word Predictor

## Project Overview

This project implements a statistical N-Gram Language Model for next-word prediction. It predicts the most likely next word based on the words entered by the user.

The project uses Unigram, Bigram and Trigram language models and provides an interactive interface built using Streamlit.

## Features

- Unigram language model
- Bigram language model
- Trigram language model
- Text preprocessing using regular expressions
- Word-frequency counting using Python Counter
- Next-word prediction using a Trigram → Bigram → Unigram fallback strategy
- Top 5 next-word predictions with probabilities
- Laplace smoothing function
- Bigram and Trigram perplexity calculation
- Interactive Streamlit interface

## Project Structure

```text
N-Gram-Next-Word-Predictor/
│
├── backend/
│   └── main.py
│
├── dataset nlp/
│   └── next_word_predictor.txt
│
├── frontend/
│   └── app.py
│
└── README.md
```

## Technologies Used

- Python
- Streamlit
- Regular Expressions (re)
- Counter and defaultdict (collections)
- pathlib
- Math

## Methodology

1. Load the text dataset.
2. Convert the text into lowercase.
3. Clean the text using regular expressions.
4. Normalize whitespace.
5. Tokenize the text into words.
6. Calculate Unigram, Bigram and Trigram frequencies.
7. Prepare word-frequency lookup structures.
8. Predict the next word using the following fallback strategy:
   - Trigram
   - Bigram
   - Unigram
9. Display the top 5 predictions with probabilities.
10. Evaluate the models using perplexity functions.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/N-Gram-Next-Word-Predictor.git
```

### 2. Navigate into the project folder

```bash
cd N-Gram-Next-Word-Predictor
```

### 3. Install Streamlit

```bash
pip install streamlit
```

### 4. Run the application

```bash
streamlit run frontend/app.py
```

### 5. Use the application

- Enter a sentence or phrase in the input box.
- Click the Predict Next Word button.
- View the predicted word, completed sentence and top 5 predictions.

## Evaluation

The project includes Bigram and Trigram perplexity calculation functions.

Accuracy, Precision, Recall and F1-score are not implemented in the current version.

## Future Improvements

- Implement a proper training and testing split.
- Integrate Laplace smoothing directly into the prediction process.
- Evaluate the models on a larger test corpus.
- Add Top-1 and Top-5 prediction accuracy.
- Explore advanced language models.

## Author

NLP Course Project – N-Gram Next-Word Prediction