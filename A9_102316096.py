# Lab Assignment 9: NLP using Python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
import string
import re
import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download(['punkt', 'stopwords', 'wordnet'])

# Q1: Text Processing
text = """Quantum computing is revolutionizing technology by harnessing quantum mechanics principles. 
Unlike classical computers that use bits, quantum computers use qubits which can exist in superposition states. 
Major companies like IBM and Google are racing to achieve quantum supremacy. 
This technology could transform fields like cryptography, drug discovery, and optimization. 
However, quantum systems are extremely sensitive to environmental interference. 
The field is still in its infancy but shows incredible promise for solving complex problems."""

print("=== Q1: Text Processing ===")
# 1. Convert to lowercase and remove punctuation
text_lower = text.lower()
text_no_punct = text_lower.translate(str.maketrans('', '', string.punctuation))

# 2. Tokenize
words = word_tokenize(text_no_punct)
sentences = sent_tokenize(text)

# 3. Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]

# 4. Word frequency distribution
fdist = FreqDist(filtered_words)
print("\nWord Frequency Distribution (Top 10):")
print(fdist.most_common(10))
fdist.plot(10, title='Top 10 Word Frequency Distribution')

# Q2: Stemming and Lemmatization
print("\n=== Q2: Stemming and Lemmatization ===")
porter = PorterStemmer()
lancaster = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

porter_stems = [porter.stem(word) for word in filtered_words]
lancaster_stems = [lancaster.stem(word) for word in filtered_words]
lemmas = [lemmatizer.lemmatize(word) for word in filtered_words]

print("\nOriginal\tPorter\t\tLancaster\tLemma")
for orig, port, lanc, lem in zip(filtered_words[:15], porter_stems[:15], lancaster_stems[:15], lemmas[:15]):
    print(f"{orig.ljust(15)}\t{port.ljust(15)}\t{lanc.ljust(15)}\t{lem}")

# Q3: Regular Expressions and Text Splitting
print("\n=== Q3: Regular Expressions ===")
words_5plus = re.findall(r'\b\w{6,}\b', text)  # 6+ letters
numbers = re.findall(r'\d+', text)
capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
alpha_only = re.findall(r'\b[a-zA-Z]+\b', text)
vowel_start = re.findall(r'\b[aeiouAEIOU][a-zA-Z]*\b', text)

print("\nWords with 6+ letters:", words_5plus)
print("Numbers found:", numbers)
print("Capitalized words:", capitalized)
print("\nAlphabetic words (first 10):", alpha_only[:10])
print("Words starting with vowel (first 10):", vowel_start[:10])

# Q4: Custom Tokenization & Regex Cleaning
print("\n=== Q4: Custom Tokenization ===")
def custom_tokenizer(text):
    tokens = re.findall(r'''
        (?:[a-zA-Z]+(?:'[a-zA-Z]+)?)  # contractions
        |(?:[a-zA-Z]+-[a-zA-Z]+)      # hyphenated words
        |(?:\d+\.\d+)                  # decimal numbers
        |(?:\d+)                       # integers
        |(?:[^\s\w])                   # special chars
    ''', text, re.VERBOSE)
    return tokens

sample_text = "Contact me at john.doe@email.com or visit https://quantum.org. Call +91 9876543210 or 123-456-7890."
tokens = custom_tokenizer(sample_text)
print("\nCustom Tokens:", tokens)

cleaned_text = re.sub(r'\S+@\S+', '<EMAIL>', sample_text)
cleaned_text = re.sub(r'https?://\S+', '<URL>', cleaned_text)
cleaned_text = re.sub(r'(?:\+\d{1,3}\s?\d{9,10}|(?:\d{3}-){2}\d{4})', '<PHONE>', cleaned_text)
print("\nCleaned Text:", cleaned_text)
