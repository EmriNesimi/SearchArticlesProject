from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = Flask(__name__)

# Load the dataset
data_file = "arXiv-DataFrame.csv"  # Replace with your dataset filename
data = pd.read_csv(data_file)

# Load the precomputed TF-IDF vectorizer and matrix
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)


# Function to rank articles by keyword
def rank_articles_by_keyword(keyword):
    # Transform the keyword into the TF-IDF space
    keyword_vector = vectorizer.transform([keyword])

    # Calculate cosine similarity
    relevance_scores = cosine_similarity(tfidf_matrix, keyword_vector).flatten()

    # Add scores to the DataFrame
    data['Relevance'] = relevance_scores

    # Sort and return articles as a dictionary
    return data.sort_values(by='Relevance', ascending=False).to_dict(orient='records')


# Home route
@app.route('/')
def home():
    return render_template('index.html')


# Search route
@app.route('/search', methods=['POST'])
def search():
    # Get the keyword from the form
    keyword = request.form.get('keyword', '').strip()

    # Rank articles by keyword
    results = rank_articles_by_keyword(keyword)

    # Limit the results to top 50
    results = results[:50]

    return render_template('search_results.html', keyword=keyword, results=results)


# Run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
