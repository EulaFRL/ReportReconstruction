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

for label, tokens in tqdm(zip(labels, processed_texts), desc="Processing data", total=len(labels)):
    label_term_frequency.update({label: len(tokens)})
    
output_path = "output/structured_term_freq.csv"
freq_df = pd.DataFrame(label_term_frequency.items(), columns=["Label", "Frequency"])
freq_df = freq_df.sort_values(by="Frequency", ascending=False)
freq_df.to_csv(output_path, index=False)

print(freq_df.head(100))