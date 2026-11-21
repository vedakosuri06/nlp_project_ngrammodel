import streamlit as st
from pathlib import Path
from collections import Counter, defaultdict
import re


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="N-Gram Next Word Predictor",
    page_icon="📝",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("📝 N-Gram Next Word Predictor")

st.write(
    "Predict the next word using Unigram, Bigram and Trigram "
    "Language Models."
)

st.divider()


# ============================================================
# DATASET PATH
# ============================================================

# Project folder:
# NLP-N Gram model
#
# Inside it:
# dataset nlp
#     next_word_predictor.txt
#
# frontend
#     app.py

project_folder = Path(__file__).resolve().parent.parent

dataset_path = (
    project_folder
    / "dataset nlp"
    / "next_word_predictor.txt"
)


# ============================================================
# CHECK DATASET
# ============================================================

if not dataset_path.exists():

    st.error("❌ Dataset file was not found.")

    st.write("Expected dataset location:")

    st.code(str(dataset_path))

    st.stop()


# ============================================================
# LOAD DATASET
# ============================================================

with open(
    dataset_path,
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


# ============================================================
# PREPROCESSING
# ============================================================

text = text.lower()

# Keep letters, numbers, apostrophes and spaces
text = re.sub(
    r"[^a-z0-9'\s]",
    " ",
    text
)

# Remove extra spaces
text = re.sub(
    r"\s+",
    " ",
    text
).strip()

# Convert text into words
words = text.split()


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

total_words = len(words)

vocabulary_size = len(set(words))


# ============================================================
# UNIGRAM MODEL
# ============================================================

unigram_counts = Counter(words)


# ============================================================
# BIGRAM MODEL
# ============================================================

bigram_counts = Counter()

for i in range(len(words) - 1):

    word1 = words[i]

    word2 = words[i + 1]

    bigram = (
        word1,
        word2
    )

    bigram_counts[bigram] += 1


# ============================================================
# TRIGRAM MODEL
# ============================================================

trigram_counts = Counter()

for i in range(len(words) - 2):

    word1 = words[i]

    word2 = words[i + 1]

    word3 = words[i + 2]

    trigram = (
        word1,
        word2,
        word3
    )

    trigram_counts[trigram] += 1


# ============================================================
# BIGRAM LOOKUP
# ============================================================

next_words_bigram = defaultdict(Counter)

for (
    word1,
    word2
), count in bigram_counts.items():

    next_words_bigram[word1][word2] = count


# ============================================================
# TRIGRAM LOOKUP
# ============================================================

next_words_trigram = defaultdict(Counter)

for (
    word1,
    word2,
    word3
), count in trigram_counts.items():

    next_words_trigram[
        (
            word1,
            word2
        )
    ][word3] = count


# ============================================================
# NEXT WORD PREDICTION FUNCTION
# ============================================================

def predict_next_word(sentence, top_n=5):

    # Convert input to lowercase
    sentence = sentence.lower()

    # Clean input
    sentence = re.sub(
        r"[^a-z0-9'\s]",
        " ",
        sentence
    )

    sentence = re.sub(
        r"\s+",
        " ",
        sentence
    ).strip()

    # Convert sentence into words
    input_words = sentence.split()


    # No input
    if len(input_words) == 0:

        return [], "No Model"


    # ========================================================
    # TRY TRIGRAM
    # ========================================================

    if len(input_words) >= 2:

        word1 = input_words[-2]

        word2 = input_words[-1]

        candidates = next_words_trigram.get(
            (
                word1,
                word2
            ),
            {}
        )

        if candidates:

            total_count = sum(
                candidates.values()
            )

            predictions = []

            for (
                word,
                count
            ) in candidates.items():

                probability = (
                    count / total_count
                )

                predictions.append(
                    (
                        word,
                        probability
                    )
                )

            # Highest probability first
            predictions.sort(
                key=lambda x: x[1],
                reverse=True
            )

            return (
                predictions[:top_n],
                "Trigram"
            )


    # ========================================================
    # TRY BIGRAM
    # ========================================================

    last_word = input_words[-1]

    candidates = next_words_bigram.get(
        last_word,
        {}
    )

    if candidates:

        total_count = sum(
            candidates.values()
        )

        predictions = []

        for (
            word,
            count
        ) in candidates.items():

            probability = (
                count / total_count
            )

            predictions.append(
                (
                    word,
                    probability
                )
            )

        predictions.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return (
            predictions[:top_n],
            "Bigram"
        )


    # ========================================================
    # UNIGRAM FALLBACK
    # ========================================================

    predictions = []

    for (
        word,
        count
    ) in unigram_counts.most_common(top_n):

        probability = (
            count / total_words
        )

        predictions.append(
            (
                word,
                probability
            )
        )

    return (
        predictions,
        "Unigram"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📊 Project Information")

    st.write("### Dataset")

    st.write(
        "Next Word Predictor Text Dataset"
    )

    st.divider()

    st.write("### Models")

    st.write("• Unigram")

    st.write("• Bigram")

    st.write("• Trigram")

    st.divider()

    st.write("### Techniques")

    st.write("• Text Preprocessing")

    st.write("• Probability Calculation")

    st.write("• N-Gram Modeling")

    st.write("• Laplace Smoothing")

    st.write("• Next-Word Prediction")

    st.divider()

    st.write("### Dataset Statistics")

    st.metric(
        "Total Words",
        f"{total_words:,}"
    )

    st.metric(
        "Vocabulary Size",
        f"{vocabulary_size:,}"
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.header("✍️ Enter Your Sentence")

st.write(
    "Type a sentence or phrase and click the button "
    "to predict the next word."
)


sentence = st.text_input(
    "Sentence",
    placeholder="Example: i am very"
)


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = st.button(
    "🔮 Predict Next Word",
    use_container_width=True
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

if predict_button:

    if sentence.strip() == "":

        st.warning(
            "⚠️ Please enter a sentence first."
        )

    else:

        predictions, model_used = predict_next_word(
            sentence
        )


        # ====================================================
        # MODEL USED
        # ====================================================

        st.subheader("🤖 Model Used")

        st.info(
            f"{model_used} Language Model"
        )


        # ====================================================
        # PREDICTED WORD
        # ====================================================

        if predictions:

            predicted_word = predictions[0][0]

            st.subheader(
                "🎯 Predicted Next Word"
            )

            st.success(
                predicted_word
            )


            # =================================================
            # COMPLETE SENTENCE
            # =================================================

            complete_sentence = (
                sentence.strip()
                + " "
                + predicted_word
            )

            st.subheader(
                "💬 Complete Sentence"
            )

            st.info(
                complete_sentence
            )


            # =================================================
            # TOP 5 PREDICTIONS
            # =================================================

            st.subheader(
                "📈 Top 5 Predictions"
            )

            for rank, (
                word,
                probability
            ) in enumerate(
                predictions,
                start=1
            ):

                st.write(
                    f"**{rank}. {word}**"
                    f" → {probability:.6f}"
                )


        else:

            st.warning(
                "No prediction available."
            )


# ============================================================
# PROJECT DETAILS
# ============================================================

st.divider()

st.header("📚 About the Project")

st.write(
    "This project implements a statistical N-Gram "
    "language model for next-word prediction."
)

st.write(
    "The model analyzes word sequences in the dataset "
    "and calculates probabilities for possible next words."
)

st.write(
    "**Unigram:** Uses individual word frequencies."
)

st.write(
    "**Bigram:** Uses the previous one word."
)

st.write(
    "**Trigram:** Uses the previous two words."
)

st.write(
    "**Laplace Smoothing:** Helps handle unseen "
    "word sequences by assigning non-zero probabilities."
)

st.divider()

st.caption(
    "N-Gram Language Model | NLP Course Project"
)