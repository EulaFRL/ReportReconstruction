import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

nltk.download('stopwords')
nltk.download('punkt')

'''
    #########################################################
    #########################################################
    Step 1: get term_freq pairs, save to /output/???_freq.csv
    #########################################################
    #########################################################
'''
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
preprocessed_texts = []
for tokens in tqdm(texts.apply(preprocess_text), desc="Processing texts"):
    preprocessed_texts.append(" ".join(tokens))
    term_freq.update(tokens)


term_freq_df = pd.DataFrame(term_freq.items(), columns=["Term", "Frequency"])
term_freq_df = term_freq_df.sort_values(by="Frequency", ascending=False)

output_path = "output/report_term_freq.csv"
term_freq_df.to_csv(output_path, index=False)

print(term_freq_df.head(100))

'''
    #########################################################
    #########################################################
    Step 2: get TF-IDF, save to /analysis_and_viz/results
    #########################################################
    #########################################################
'''
print("Calculating TF-IDF...")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)

tfidf_scores = tfidf_matrix.sum(axis=0).A1
terms = vectorizer.get_feature_names_out()

tfidf_summary_df = pd.DataFrame({
    "Term": terms,
    "TF-IDF Score": tfidf_scores
}).sort_values(by="TF-IDF Score", ascending=False)

tfidf_output_path = "analysis_and_viz/results/report_tfidf.csv"
tfidf_summary_df.to_csv(tfidf_output_path, index=False)

print(f"TF-IDF results saved to {tfidf_output_path}")
print("Top terms by TF-IDF:")
print(tfidf_summary_df.head(10))