import re

# lightweight stopword list (no nltk required)
STOPWORDS = {
"a","an","the","and","or","but","if","while","with","to","from","of","at","by",
"for","in","on","out","over","under","again","further","then","once","here","there",
"all","any","both","each","few","more","most","other","some","such","no","nor","not",
"only","own","same","so","than","too","very","can","will","just","don","should","now",
"is","am","are","was","were","be","been","being","have","has","had","do","does","did"
}

def clean_text(text):
    # lowercase
    text = text.lower()

    # remove special chars + numbers
    text = re.sub(r'[^a-z\s]', ' ', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # remove stopwords
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]

    return " ".join(words)
#I intentionally replaced NLTK with a custom lightweight preprocessing pipeline because cloud environments often restrict downloads and heavy dependencies. This made my model faster, portable, and production-ready.