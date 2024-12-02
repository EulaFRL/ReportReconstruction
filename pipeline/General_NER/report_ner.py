import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm

nltk.download('stopwords')
nltk.download('punkt')


# load the data, TEXT portion only
df = pd.read_csv("data/lung_cancer_notes.csv")
texts = df["TEXT"]

'''
    Helper function for preprocessing the note
    - lowercase conversion, special characters replacement
    - tokenization (using punkt)
'''
def preprocess_text(text):
    tqdm_desc = f"Processing text"
    tokens = []
    for word in tqdm(re.sub(r"[^a-z\s]", "", text.lower()).split(), desc=tqdm_desc):
        if word not in stopwords.words("english"):
            tokens.append(word)
    return tokens

'''
    Collect frequency in tokenized text, save in csv
'''
term_freq = Counter()
for tokens in tqdm(texts.apply(preprocess_text), desc="Processing texts"):
    term_freq.update(tokens)


term_freq_df = pd.DataFrame(term_freq.items(), columns=["Term", "Frequency"])
term_freq_df = term_freq_df.sort_values(by="Frequency", ascending=False)

output_path = "output/report_term_freq.csv"
term_freq_df.to_csv(output_path, index=False)

print(term_freq_df.head(100))