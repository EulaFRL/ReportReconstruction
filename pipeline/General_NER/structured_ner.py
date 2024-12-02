import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from tqdm import tqdm
nltk.download('stopwords')
nltk.download('punkt')

# load the data, TEXT portion only
df = pd.read_csv("data/NLST_concatenated.csv")
labels = df["Label"]  # attribute names
texts = df["Concatenated"]  # corresponding concatenated data

'''
    Helper function for preprocessing the note
    - lowercase conversion, special characters replacement
    - tokenization (using punkt)
'''
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return tokens

processed_texts = texts.apply(preprocess_text)


'''
    Helper function for classifying non medical terms.
    TODO: the medical_terms is an examplary corpus that should be replaced with medical
    corpus later.
'''

label_term_frequency = Counter()
preprocessed_texts = []
for label, tokens in tqdm(zip(labels, processed_texts), desc="Processing data", total=len(labels)):
    preprocessed_texts.append(" ".join(tokens))
    label_term_frequency.update({label: len(tokens)})
    
output_path = "output/structured_term_freq.csv"
freq_df = pd.DataFrame(label_term_frequency.items(), columns=["Label", "Frequency"])
freq_df = freq_df.sort_values(by="Frequency", ascending=False)
freq_df.to_csv(output_path, index=False)

print(freq_df.head(100))

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

tfidf_output_path = "analysis_and_viz/results/structured_tfidf.csv"
tfidf_summary_df.to_csv(tfidf_output_path, index=False)

print(f"TF-IDF results saved to {tfidf_output_path}")
print("Top terms by TF-IDF:")
print(tfidf_summary_df.head(10))