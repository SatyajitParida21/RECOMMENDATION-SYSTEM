 import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_movie_recommender():
    # Sample movie dataset (YOU ALSO CAN ADD MOVIE NAME AND ABOUT ITS DESCRIPTION TO FILTER OUT THE CONTENT)
    movies = pd.DataFrame({
        'title': [ 
            "3 Idiots", "Zindagi Na Milegi Dobara", "Dilwale Dulhania Le Jayenge", 
            "Kabir Singh", "Bajrangi Bhaijaan"
        ],
        'description': [
            'A comedy-drama about three friends navigating college life, societal pressure, and the importance of following one’s passion.',
            'A road trip movie about three friends who embark on a journey of self-discovery and overcoming fears.',
            'A classic romantic film where Raj tries to win Simran’s love and her family’s approval.',
            'A passionate love story of a brilliant but self-destructive surgeon struggling with heartbreak.',
            'A heartwarming tale of a man helping a mute Pakistani girl reunite with her family.'
        ]
    })
    
    # Convert text to feature vectors (INTERPRET)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(movies['description'])
    
    # Compute similarity scores
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)


    
    return movies, similarity_matrix

def recommend_movies(movie_title, movies, similarity_matrix):
    if not (movies['title'] == movie_title).any():
        return f"Movie '{movie_title}' not found. Please enter a valid title from the dataset.EITHER ENTER THE MOVIE NAME AND ITS DESCRIPTION IN DATASET"
    
    # Get the index of the given movie
    idx = movies[movies['title'] == movie_title].index[0]
    
    # Get similarity scores and sort them
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 3 recommendations (excluding the movie itself)
    recommended_indices = [i[0] for i in similarity_scores[1:4]]
    recommended_movies = movies.iloc[recommended_indices]['title'].tolist()
    
    return recommended_movies

# Example usage
movies, similarity_matrix = create_movie_recommender()
movie_title = "Kabir Singh"  # Change this to a movie from the dataset
recommendations = recommend_movies(movie_title, movies, similarity_matrix)
print(f"Movies similar to '{movie_title}': {recommendations}")
