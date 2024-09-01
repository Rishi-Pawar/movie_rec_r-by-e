from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances

app = Flask(__name__)

# Load movie data
movies = pd.read_csv('movies.csv', encoding='latin1')
genres = pd.read_csv('movie_genres.csv', encoding='latin1')
movies_df = pd.read_csv('movies.csv', encoding='latin1')
movie_genres_df = pd.read_csv('movie_genres.csv', encoding='latin1')
# Merge movie data with genres
movies = movies.merge(genres, left_on='id', right_on='movieID')

# Initialize CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')

# Fit and transform the data
vector = cv.fit_transform(movies['genre']).toarray()

# Compute Jaccard similarity
similarity = pairwise_distances(vector, metric='jaccard')

# Define similarity threshold and epsilon for r-by-e model
theta = 0.3
epsilon = 0.1

#----------------------------------------------
# Functions for the content-based model
def recommend_movies_by_genres(selected_genres, threshold=0.2, num_recommendations=5):
    selected_genres_str = ' '.join(selected_genres)
    selected_vector = cv.transform([selected_genres_str]).toarray()
    
    # Compute similarity between the selected genres and all movies
    genre_similarity = pairwise_distances(selected_vector, vector, metric='jaccard').flatten()
    
    # Get indices of the most similar movies, remove duplicates
    similar_movies_indices = np.argsort(genre_similarity)
    recommended_movies = []
    seen_titles = set()
    
    for idx in similar_movies_indices:
        movie = movies.iloc[idx]
        if movie['title'] not in seen_titles:
            recommended_movies.append({'id': movie['id'], 'title': movie['title']})
            seen_titles.add(movie['title'])
        if len(recommended_movies) >= num_recommendations:
            break
    
    return recommended_movies

#-----------------------------------------------------------------
#-----------------------------------------------------------------

def similarity(genres1, genres2):
    return len(set(genres1).intersection(genres2)) / len(genres1)

def reward(candidate_movie_id, profile_movies, predecessor_movie_id, theta, epsilon):
    candidate_features = set(movie_genres_df[movie_genres_df['movieID'] == candidate_movie_id]['genre'])
    predecessor_features = set(movie_genres_df[movie_genres_df['movieID'] == predecessor_movie_id]['genre'])
    covered_features = set()

    for movie in profile_movies:
        movie_features = set(movie_genres_df[movie_genres_df['movieID'] == movie]['genre'])
        covered_features.update(candidate_features.intersection(movie_features))

    fi_minus_covered = len(candidate_features - covered_features)
    fi = len(candidate_features)
    fp_minus_covered = len(predecessor_features - covered_features)
    fp = len(predecessor_features)

    if fi == 0 or fp == 0:
        return 0

    reward_value = (fi_minus_covered / fi) + (fp_minus_covered / fp)
    return reward_value

def get_movie_title(movie_id):
    return movies_df[movies_df['id'] == movie_id]['title'].iloc[0]

def get_movie_features(movie_id):
    return movie_genres_df[movie_genres_df['movieID'] == movie_id]['genre'].tolist()

def generate_explanation_chain(candidate_movie_id, profile_movies, theta, epsilon):
    explanation_chain = []
    for movie_id in profile_movies:
        rwd = reward(candidate_movie_id, profile_movies, movie_id, theta, epsilon)
        if rwd > epsilon:
            explanation_chain.append(movie_id)
    return explanation_chain

def scoring(chain, candidate_movie_id, profile_movies, theta, epsilon):
    sum_rwds = sum(reward(candidate_movie_id, profile_movies, movie_id, theta, epsilon) for movie_id in chain)
    diversity_penalty = sum(len(set(get_movie_features(candidate_movie_id)).difference(get_movie_features(movie_id))) for movie_id in chain)
    score = (sum_rwds / (len(chain) + 1)) + (diversity_penalty / (len(chain) + 1))
    return score

def select_chains(chains, n):
    chains.sort(key=lambda x: x[2], reverse=True)
    selected_chains = chains[:n]
    return selected_chains

def recommend_movies_rbye(user_profile, n_recommendations, theta, epsilon):
    candidate_movies = movies_df['id'].tolist()
    recommended_movies = []
    for candidate_movie_id in candidate_movies:
        if candidate_movie_id not in user_profile:  # Exclude movies already in user's profile
            explanation_chain = generate_explanation_chain(candidate_movie_id, user_profile, theta, epsilon)
            if explanation_chain:
                score = scoring(explanation_chain, candidate_movie_id, user_profile, theta, epsilon)
                recommended_movies.append((candidate_movie_id, explanation_chain, score))

    top_n_chains = select_chains(recommended_movies, n_recommendations)
    return top_n_chains
#-----------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_genres = request.form.getlist('genres')
        if len(selected_genres) < 3:
            return render_template('index.html', genres=unique_genres, error="Please select at least 3 genres.")
        
        recommendations = recommend_movies_by_genres(selected_genres)
        return render_template('content_recommendations.html', recommendations=recommendations, genres=selected_genres)
    
    return render_template('index.html', genres=unique_genres)

@app.route('/select_content_recommendations', methods=['POST'])
def select_content_recommendations():
    selected_movies = request.form.getlist('selected_movies')
    user_profile = list(map(int, selected_movies))
    return render_template('rbye_recommendations.html', user_profile=user_profile)

@app.route('/rbye_recommendations', methods=['POST'])
def rbye_recommendations():
    user_profile = list(map(int, request.form.getlist('user_profile')))
    selected_movies = request.form.getlist('selected_movies')
    user_profile.extend(list(map(int, selected_movies)))

    recommendations = recommend_movies_rbye(user_profile, n_recommendations=5, theta=theta, epsilon=epsilon)
    
    results = []
    for movie_id, explanation_chain, score in recommendations:
        movie_title = get_movie_title(movie_id)
        explanation_chain_info = [(get_movie_title(movie_id), get_movie_features(movie_id)) for movie_id in explanation_chain]
        results.append({
            'movie_title': movie_title,
            'explanation_chain': explanation_chain_info,
            'score': score
        })
    
    return render_template('rbye_recommendations.html', user_profile=user_profile, results=results)

if __name__ == '__main__':
    unique_genres = sorted(movies['genre'].unique())
    app.run(debug=True)


# from flask import Flask, render_template, request
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import pairwise_distances

# app = Flask(__name__)

# # Load movie data
# movies = pd.read_csv('movies.csv', encoding='latin1')
# genres = pd.read_csv('movie_genres.csv', encoding='latin1')
# movies_df = pd.read_csv('movies.csv', encoding='latin1')
# movie_genres_df = pd.read_csv('movie_genres.csv', encoding='latin1')
# # Merge movie data with genres
# movies = movies.merge(genres, left_on='id', right_on='movieID')

# # Initialize CountVectorizer
# cv = CountVectorizer(max_features=5000, stop_words='english')

# # Fit and transform the data
# vector = cv.fit_transform(movies['genre']).toarray()

# # Compute Jaccard similarity
# similarity = pairwise_distances(vector, metric='jaccard')

# # Define similarity threshold and epsilon for r-by-e model
# theta = 0.3
# epsilon = 0.1

# #----------------------------------------------
# # Functions for the content-based model
# def recommend_movies_by_genres(selected_genres, threshold=0.2, num_recommendations=5):
#     selected_genres_str = ' '.join(selected_genres)
#     selected_vector = cv.transform([selected_genres_str]).toarray()
    
#     # Compute similarity between the selected genres and all movies
#     genre_similarity = pairwise_distances(selected_vector, vector, metric='jaccard').flatten()
    
#     # Get indices of the most similar movies, remove duplicates
#     similar_movies_indices = np.argsort(genre_similarity)
#     recommended_movies = []
#     seen_titles = set()
    
#     for idx in similar_movies_indices:
#         movie = movies.iloc[idx]
#         if movie['title'] not in seen_titles:
#             recommended_movies.append({'id': movie['id'], 'title': movie['title']})
#             seen_titles.add(movie['title'])
#         if len(recommended_movies) >= num_recommendations:
#             break
    
#     return recommended_movies

# #-----------------------------------------------------------------
# #-----------------------------------------------------------------

# def similarity(genres1, genres2):
#     return len(set(genres1).intersection(genres2)) / len(genres1)

# def reward(candidate_movie_id, profile_movies, predecessor_movie_id, theta, epsilon):
#     candidate_features = set(movie_genres_df[movie_genres_df['movieID'] == candidate_movie_id]['genre'])
#     predecessor_features = set(movie_genres_df[movie_genres_df['movieID'] == predecessor_movie_id]['genre'])
#     covered_features = set()

#     for movie in profile_movies:
#         movie_features = set(movie_genres_df[movie_genres_df['movieID'] == movie]['genre'])
#         covered_features.update(candidate_features.intersection(movie_features))

#     fi_minus_covered = len(candidate_features - covered_features)
#     fi = len(candidate_features)
#     fp_minus_covered = len(predecessor_features - covered_features)
#     fp = len(predecessor_features)

#     if fi == 0 or fp == 0:
#         return 0

#     reward_value = (fi_minus_covered / fi) + (fp_minus_covered / fp)
#     return reward_value

# def get_movie_title(movie_id):
#     return movies_df[movies_df['id'] == movie_id]['title'].iloc[0]

# def get_movie_features(movie_id):
#     return movie_genres_df[movie_genres_df['movieID'] == movie_id]['genre'].tolist()

# def generate_explanation_chain(candidate_movie_id, profile_movies, theta, epsilon):
#     explanation_chain = []
#     for movie_id in profile_movies:
#         rwd = reward(candidate_movie_id, profile_movies, movie_id, theta, epsilon)
#         if rwd > epsilon:
#             explanation_chain.append(movie_id)
#     return explanation_chain

# def scoring(chain, candidate_movie_id, profile_movies, theta, epsilon):
#     sum_rwds = sum(reward(candidate_movie_id, profile_movies, movie_id, theta, epsilon) for movie_id in chain)
#     diversity_penalty = sum(len(set(get_movie_features(candidate_movie_id)).difference(get_movie_features(movie_id))) for movie_id in chain)
#     score = (sum_rwds / (len(chain) + 1)) + (diversity_penalty / (len(chain) + 1))
#     return score

# def select_chains(chains, n):
#     chains.sort(key=lambda x: x[2], reverse=True)
#     selected_chains = chains[:n]
#     return selected_chains

# def recommend_movies_rbye(user_profile, n_recommendations, theta, epsilon):
#     candidate_movies = movies_df['id'].tolist()
#     recommended_movies = []
#     for candidate_movie_id in candidate_movies:
#         if candidate_movie_id not in user_profile:  # Exclude movies already in user's profile
#             explanation_chain = generate_explanation_chain(candidate_movie_id, user_profile, theta, epsilon)
#             if explanation_chain:
#                 score = scoring(explanation_chain, candidate_movie_id, user_profile, theta, epsilon)
#                 recommended_movies.append((candidate_movie_id, explanation_chain, score))

#     top_n_chains = select_chains(recommended_movies, n_recommendations)
#     return top_n_chains

# #-----------------------------------------------------
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         selected_genres = request.form.getlist('genres')
#         if len(selected_genres) < 3:
#             return render_template('index.html', genres=unique_genres, error="Please select at least 3 genres.")
        
#         recommendations = recommend_movies_by_genres(selected_genres)
#         return render_template('content_recommendations.html', recommendations=recommendations, genres=selected_genres)
    
#     return render_template('index.html', genres=unique_genres)

# @app.route('/select_content_recommendations', methods=['POST'])
# def select_content_recommendations():
#     selected_movies = request.form.getlist('selected_movies')
#     user_profile = list(map(int, selected_movies))
#     recommendations = recommend_movies_rbye(user_profile, n_recommendations=5, theta=theta, epsilon=epsilon)
    
#     results = []
#     for movie_id, explanation_chain, score in recommendations:
#         movie_title = get_movie_title(movie_id)
#         explanation_chain_info = [(get_movie_title(movie_id), get_movie_features(movie_id)) for movie_id in explanation_chain]
#         results.append({
#             'movie_title': movie_title,
#             'explanation_chain': explanation_chain_info,
#             'score': score
#         })
    
#     return render_template('rbye_recommendations.html', user_profile=user_profile, results=results)

# @app.route('/rbye_recommendations', methods=['POST'])
# def rbye_recommendations():
#     user_profile = list(map(int, request.form.getlist('user_profile')))
#     selected_movies = request.form.getlist('selected_movies')
#     user_profile.extend(list(map(int, selected_movies)))

#     recommendations = recommend_movies_rbye(user_profile, n_recommendations=5, theta=theta, epsilon=epsilon)
    
#     results = []
#     for movie_id, explanation_chain, score in recommendations:
#         movie_title = get_movie_title(movie_id)
#         explanation_chain_info = [(get_movie_title(movie_id), get_movie_features(movie_id)) for movie_id in explanation_chain]
#         results.append({
#             'movie_title': movie_title,
#             'explanation_chain': explanation_chain_info,
#             'score': score
#         })
    
#     return render_template('rbye_recommendations.html', user_profile=user_profile, results=results)

# if __name__ == '__main__':
#     unique_genres = sorted(movies['genre'].unique())
#     app.run(debug=True)
