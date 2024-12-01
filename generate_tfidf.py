import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the dataset
data_file = "arXiv-DataFrame.csv"  # Replace with your dataset filename
data = pd.read_csv(data_file)

# Combine Title and Summary columns for TF-IDF calculation
data['Combined_Text'] = data['Title'] + " " + data['Summary']

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Compute the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(data['Combined_Text'])

# Save the vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save the TF-IDF matrix
with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

print("TF-IDF vectorizer and matrix saved successfully!")
