from pathlib import Path
from collections import Counter, defaultdict
import re
import math


# ============================================================
# 1. LOAD DATASET
# ============================================================

# Find the main project folder
project_folder = Path(__file__).resolve().parent.parent

# Dataset path
dataset_path = (
    project_folder
    / "dataset nlp"
    / "next_word_predictor.txt"
)

print("Looking for dataset at:")
print(dataset_path)


# Check dataset
if not dataset_path.exists():

    print("\nERROR: Dataset file was not found!")

    print("\nExpected folder structure:")
    print("NLP-N Gram model")
    print("│")
    print("├── backend")
    print("│   └── main.py")
    print("│")
    print("└── dataset nlp")
    print("    └── next_word_predictor.txt")

    exit()


# Read dataset
with open(
    dataset_path,
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


print("\nDataset loaded successfully!")

print(
    "Total characters:",
    len(text)
)


# ============================================================
# 2. PREPROCESSING
# ============================================================

# Convert text to lowercase
text = text.lower()


# Keep:
# letters
# numbers
# apostrophes
# spaces
#
# This keeps words such as:
# don't
# i'm
# you're
# that's

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


print(
    "\n========== PREPROCESSING RESULTS =========="
)

print(
    "Total words:",
    len(words)
)

print(
    "\nFirst 50 words:"
)

print(
    words[:50]
)


# ============================================================
# 3. UNIGRAM MODEL
# ============================================================

# A unigram contains one word.

unigram_counts = Counter(words)

total_words = len(words)

vocabulary_size = len(unigram_counts)


print(
    "\n========== UNIGRAM MODEL =========="
)

print(
    "Total words:",
    total_words
)

print(
    "Unique words:",
    vocabulary_size
)


print(
    "\nMost common 20 words:"
)

for word, count in unigram_counts.most_common(20):

    print(
        word,
        "->",
        count
    )


# ============================================================
# 4. UNIGRAM PROBABILITY
# ============================================================

# Formula:
#
# P(word) = Count(word) / Total words


def unigram_probability(word):

    count = unigram_counts.get(
        word,
        0
    )

    if total_words == 0:

        return 0

    return count / total_words


# Example
test_word = "the"


print(
    "\n========== UNIGRAM PROBABILITY =========="
)

print(
    "Word:",
    test_word
)

print(
    "Count:",
    unigram_counts.get(
        test_word,
        0
    )
)

print(
    "Probability:",
    unigram_probability(
        test_word
    )
)


# ============================================================
# 5. BIGRAM MODEL
# ============================================================

# A bigram contains two consecutive words.
#
# Example:
#
# natural language
# language processing


bigram_counts = Counter()


for i in range(
    len(words) - 1
):

    word1 = words[i]

    word2 = words[i + 1]

    bigram = (
        word1,
        word2
    )

    bigram_counts[bigram] += 1


print(
    "\n========== BIGRAM MODEL =========="
)

print(
    "Total bigrams:",
    len(words) - 1
)

print(
    "Unique bigrams:",
    len(bigram_counts)
)


print(
    "\nMost common 20 bigrams:"
)

for bigram, count in bigram_counts.most_common(20):

    print(
        bigram,
        "->",
        count
    )


# ============================================================
# 6. BIGRAM PROBABILITY
# ============================================================

# Formula:
#
# P(word2 | word1)
#
# =
#
# Count(word1, word2)
# -------------------
# Count(word1)


def bigram_probability(
    word1,
    word2
):

    bigram_count = bigram_counts.get(
        (
            word1,
            word2
        ),
        0
    )

    previous_word_count = unigram_counts.get(
        word1,
        0
    )

    if previous_word_count == 0:

        return 0

    return (
        bigram_count
        /
        previous_word_count
    )


# Example
print(
    "\n========== BIGRAM PROBABILITY =========="
)

print(
    "P(the | of)"
)

print(
    "Bigram count:",
    bigram_counts.get(
        (
            "of",
            "the"
        ),
        0
    )
)

print(
    "Previous word count:",
    unigram_counts.get(
        "of",
        0
    )
)

print(
    "Probability:",
    bigram_probability(
        "of",
        "the"
    )
)


# ============================================================
# 7. TRIGRAM MODEL
# ============================================================

# A trigram contains three consecutive words.
#
# Example:
#
# natural language processing


trigram_counts = Counter()


for i in range(
    len(words) - 2
):

    word1 = words[i]

    word2 = words[i + 1]

    word3 = words[i + 2]

    trigram = (
        word1,
        word2,
        word3
    )

    trigram_counts[trigram] += 1


print(
    "\n========== TRIGRAM MODEL =========="
)

print(
    "Total trigrams:",
    len(words) - 2
)

print(
    "Unique trigrams:",
    len(trigram_counts)
)


print(
    "\nMost common 20 trigrams:"
)

for trigram, count in trigram_counts.most_common(20):

    print(
        trigram,
        "->",
        count
    )


# ============================================================
# 8. TRIGRAM PROBABILITY
# ============================================================

# Formula:
#
# P(word3 | word1, word2)
#
# =
#
# Count(word1, word2, word3)
# --------------------------
# Count(word1, word2)


def trigram_probability(
    word1,
    word2,
    word3
):

    trigram_count = trigram_counts.get(
        (
            word1,
            word2,
            word3
        ),
        0
    )

    previous_bigram_count = bigram_counts.get(
        (
            word1,
            word2
        ),
        0
    )

    if previous_bigram_count == 0:

        return 0

    return (
        trigram_count
        /
        previous_bigram_count
    )


# Example
print(
    "\n========== TRIGRAM PROBABILITY =========="
)

print(
    "P(of | the, echoes)"
)

print(
    "Trigram count:",
    trigram_counts.get(
        (
            "the",
            "echoes",
            "of"
        ),
        0
    )
)

print(
    "Previous bigram count:",
    bigram_counts.get(
        (
            "the",
            "echoes"
        ),
        0
    )
)

print(
    "Probability:",
    trigram_probability(
        "the",
        "echoes",
        "of"
    )
)


# ============================================================
# 9. LAPLACE SMOOTHING
# ============================================================

# Add-one / Laplace smoothing:
#
# P(word3 | word1, word2)
#
# =
#
# Count(word1, word2, word3) + 1
# --------------------------------
# Count(word1, word2) + V
#
# V = vocabulary size


def trigram_probability_smoothed(
    word1,
    word2,
    word3
):

    trigram_count = trigram_counts.get(
        (
            word1,
            word2,
            word3
        ),
        0
    )

    previous_bigram_count = bigram_counts.get(
        (
            word1,
            word2
        ),
        0
    )

    probability = (
        trigram_count + 1
    ) / (
        previous_bigram_count
        +
        vocabulary_size
    )

    return probability


print(
    "\n========== LAPLACE SMOOTHING =========="
)

print(
    "Word sequence: the echoes of"
)

print(
    "Original count:",
    trigram_counts.get(
        (
            "the",
            "echoes",
            "of"
        ),
        0
    )
)

print(
    "Previous bigram count:",
    bigram_counts.get(
        (
            "the",
            "echoes"
        ),
        0
    )
)

print(
    "Vocabulary size:",
    vocabulary_size
)

print(
    "Smoothed probability:",
    trigram_probability_smoothed(
        "the",
        "echoes",
        "of"
    )
)


# ============================================================
# 10. CREATE BIGRAM LOOKUP
# ============================================================

# Used for quickly finding:
#
# word -> possible next words


next_words_bigram = defaultdict(
    Counter
)


for (
    word1,
    word2
), count in bigram_counts.items():

    next_words_bigram[
        word1
    ][word2] = count


# ============================================================
# 11. CREATE TRIGRAM LOOKUP
# ============================================================

# Used for quickly finding:
#
# (word1, word2)
#       ↓
# possible next words


next_words_trigram = defaultdict(
    Counter
)


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
# 12. NEXT WORD PREDICTION
# ============================================================

def predict_next_word(
    sentence,
    top_n=5
):

    # --------------------------------------------------------
    # CLEAN INPUT
    # --------------------------------------------------------

    sentence = sentence.lower()

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

    input_words = sentence.split()


    # Empty input
    if len(input_words) == 0:

        return [], "No input"


    # ========================================================
    # TRIGRAM PREDICTION
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
                next_word,
                count
            ) in candidates.items():

                probability = (
                    count
                    /
                    total_count
                )

                predictions.append(
                    (
                        next_word,
                        probability
                    )
                )


            predictions.sort(
                key=lambda x: x[1],
                reverse=True
            )


            return (
                predictions[:top_n],
                "Trigram"
            )


    # ========================================================
    # BIGRAM PREDICTION
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
            next_word,
            count
        ) in candidates.items():

            probability = (
                count
                /
                total_count
            )

            predictions.append(
                (
                    next_word,
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
    ) in unigram_counts.most_common(
        top_n
    ):

        probability = (
            count
            /
            total_words
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
# 13. BIGRAM PERPLEXITY
# ============================================================

def calculate_bigram_perplexity(
    sentence
):

    sentence = sentence.lower()

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

    test_words = sentence.split()


    # At least two words required
    if len(test_words) < 2:

        return None


    log_probability = 0

    count = 0


    for i in range(
        len(test_words) - 1
    ):

        word1 = test_words[i]

        word2 = test_words[i + 1]


        # Get probability
        probability = bigram_probability(
            word1,
            word2
        )


        # If unseen, apply smoothing
        if probability == 0:

            probability = (
                1
                /
                (
                    unigram_counts.get(
                        word1,
                        0
                    )
                    +
                    vocabulary_size
                )
            )


        log_probability += math.log(
            probability
        )

        count += 1


    # Perplexity formula
    perplexity = math.exp(
        -log_probability / count
    )


    return perplexity


# ============================================================
# 14. TRIGRAM PERPLEXITY
# ============================================================

def calculate_trigram_perplexity(
    sentence
):

    sentence = sentence.lower()

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

    test_words = sentence.split()


    # At least three words required
    if len(test_words) < 3:

        return None


    log_probability = 0

    count = 0


    for i in range(
        len(test_words) - 2
    ):

        word1 = test_words[i]

        word2 = test_words[i + 1]

        word3 = test_words[i + 2]


        # Get probability
        probability = trigram_probability(
            word1,
            word2,
            word3
        )


        # If unseen, apply smoothing
        if probability == 0:

            probability = (
                1
                /
                (
                    bigram_counts.get(
                        (
                            word1,
                            word2
                        ),
                        0
                    )
                    +
                    vocabulary_size
                )
            )


        log_probability += math.log(
            probability
        )

        count += 1


    # Perplexity formula
    perplexity = math.exp(
        -log_probability / count
    )


    return perplexity


# ============================================================
# 15. PERPLEXITY TEST
# ============================================================

test_sentence = "it was a beautiful day"


print(
    "\n========== PERPLEXITY TEST =========="
)

print(
    "Test sentence:",
    test_sentence
)


bigram_ppl = calculate_bigram_perplexity(
    test_sentence
)


trigram_ppl = calculate_trigram_perplexity(
    test_sentence
)


print(
    "Bigram Perplexity:",
    round(
        bigram_ppl,
        4
    )
)


print(
    "Trigram Perplexity:",
    round(
        trigram_ppl,
        4
    )
)


# ============================================================
# 16. INTERACTIVE NEXT WORD PREDICTOR
# ============================================================

print("\n")

print("=" * 60)

print(
    "             N-GRAM NEXT WORD PREDICTOR"
)

print("=" * 60)


print(
    "\nYou can enter any sentence."
)

print(
    "The model will predict the next word."
)

print(
    "Type 'exit' to stop."
)


# ============================================================
# 17. MAIN LOOP
# ============================================================

while True:

    sentence = input(
        "\nEnter your sentence: "
    )


    # Exit
    if sentence.lower().strip() == "exit":

        print(
            "\nProgram ended."
        )

        break


    # Predict
    predictions, model_used = predict_next_word(
        sentence,
        top_n=5
    )


    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    print(
        "\nInput:",
        sentence
    )


    # --------------------------------------------------------
    # MODEL USED
    # --------------------------------------------------------

    print(
        "Model used:",
        model_used
    )


    # --------------------------------------------------------
    # TOP 5 PREDICTIONS
    # --------------------------------------------------------

    print(
        "\nTop 5 predictions:"
    )


    for (
        word,
        probability
    ) in predictions:

        print(
            word,
            "->",
            round(
                probability,
                6
            )
        )


    # --------------------------------------------------------
    # BEST PREDICTION
    # --------------------------------------------------------

    if predictions:

        predicted_word = predictions[0][0]


        print(
            "\nPredicted next word:",
            predicted_word
        )


        # Add prediction to input
        complete_sentence = (
            sentence.strip()
            +
            " "
            +
            predicted_word
        )


        print(
            "\nComplete sentence:"
        )


        print(
            complete_sentence
        )


    else:

        print(
            "\nNo prediction available."
        )