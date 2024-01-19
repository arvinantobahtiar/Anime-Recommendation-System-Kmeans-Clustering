from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from fuzzywuzzy import process
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Load the saved models
kmeans_model1 = joblib.load('pkl/kmeans_model1.pkl')

df = pd.read_csv('Anime-preprocessed.csv')

# Custom tokenizer to split genres based on commas
def custom_tokenizer(text):
    return text.split(', ')

# TF-IDF vectorization for 'Genres' column
vectorizer_genres = TfidfVectorizer(stop_words='english', tokenizer=custom_tokenizer)
genres_tfidf = vectorizer_genres.fit_transform(df['Genres'])
genres_tfidf_df = pd.DataFrame(genres_tfidf.toarray(), columns=vectorizer_genres.get_feature_names_out())

# List of top 10 genres by popularity in animes
top_genres_list = ['action', 'adventure', 'romance', 'comedy', 'magic', 'sci fi', 'seinen', 'shounen', 'sports', 'fantasy']

# Filter columns based on the top genres list
genres_tfidf_df = genres_tfidf_df[top_genres_list]

# Separate data for Model 1 (Genres)
data1 = genres_tfidf_df.values

# Standardize the data for Model 1
scaler_model1 = StandardScaler()
scaled_data_model1 = scaler_model1.fit_transform(data1)

# Add the 'Cluster_model1' column to the DataFrame
model1_labels = kmeans_model1.predict(scaled_data_model1)
df['Cluster_model1'] = model1_labels

# Flask endpoint to receive anime title and return recommendations
@app.route('/')
def Home():
    return render_template('index.html')

# Function to recommend anime based on input title
def recommend_anime(input_title, cluster_column, df):
    # Use fuzzy matching to find the closest matching anime title
    result = process.extractOne(input_title, df['Name'])
    match, score = result[0], result[1]

    # Retrieve the cluster of the closest matching anime
    closest_anime_id = df[df['Name'] == match].index
    closest_anime_cluster = df.loc[closest_anime_id, cluster_column].values[0]

    # Filter recommendations from the same cluster as the closest matching anime
    cluster_mask = (df[cluster_column] == closest_anime_cluster)
    cluster_animes = df[cluster_mask]['Name'].tolist()

    # Return 10 recommendations from the same cluster
    recommendations = random.sample(cluster_animes, min(10, len(cluster_animes)))
    
    # Include the matched anime in the response
    response = {'matched_anime': match, 'recommendations': recommendations}
    return response

# Flask endpoint to receive anime title and return recommendations
@app.route('/recommend', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    input_title = data.get('title')
    response = recommend_anime(input_title, 'Cluster_model1', df) 
    return jsonify(response)

if __name__=="__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')