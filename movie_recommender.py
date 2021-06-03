# Dependencies
import pandas as pd
import numpy as np

# Loading in Movies CSV
movie_list = pd.read_csv("tmdb_5000_movies.csv")
movies_db = pd.DataFrame(movie_list)
# Loading in Credits CSV
ratings_list = pd.read_csv("tmdb_5000_credits.csv")
ratings_db = pd.DataFrame(ratings_list)
# Merging the Data sets
movies_db = movies_db.rename(columns = {'id':'movie_id'})
ratings_db.columns = ['movie_id','title','cast','crew']
movie_ratings = movies_db.merge(ratings_db, on="movie_id")
#Creating variables for the avg vote and count
avg_rating = movie_ratings['vote_average'].mean()
rating_count = movie_ratings['vote_count'].count()
# Created a variable for the minimum amount of ratings needed for it to be included
m= movie_ratings['vote_count'].quantile(0.9)
filtered_movies = movie_ratings.copy().loc[movie_ratings['vote_count'] >= m]
#Created the weights for the variables
def weighted_rating(x, m=m, C=avg_rating):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)
# Define a new feature 'score' and calculate its value with `weighted_rating()`
filtered_movies['score'] = filtered_movies.apply(weighted_rating, axis=1)
#Sort movies based on score calculated above
filtered_movies = filtered_movies.sort_values('score', ascending=False)

#Print the top 15 movies
filtered_movies[['title_x', 'vote_count', 'vote_average', 'score']].head(10)
#Created Popular Movies Chart
pop= movie_ratings.sort_values('popularity', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title_x'].head(6),pop['popularity'].head(6), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
#Showing what the 'overview' column looks like
movie_ratings['overview'].head(5)

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
movie_ratings['overview'] = movie_ratings['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(movie_ratings['overview'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#Construct a reverse map of indices and movie titles
indices = pd.Series(movie_ratings.index, index=movie_ratings['title_x']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title_x, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title_x]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movie_ratings['title_x'].iloc[movie_indices]
# Parse the stringified features into their corresponding python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    movie_ratings[feature] = movie_ratings[feature].apply(literal_eval)
# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []
# Define new director, cast, genres and keywords features that are in a suitable form.
movie_ratings['director'] = movie_ratings['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    movie_ratings[feature] = movie_ratings[feature].apply(get_list)
# Print the new features of the first 3 films
print(movie_ratings[['title_x', 'cast', 'director', 'keywords', 'genres']].head(3))

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    movie_ratings[feature] = movie_ratings[feature].apply(clean_data)
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
movie_ratings['soup'] = movie_ratings.apply(create_soup, axis=1)
# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(movie_ratings['soup'])
# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
# Reset index of our main DataFrame and construct reverse mapping as before
movie_ratings = movie_ratings.reset_index()
indices = pd.Series(movie_ratings.index, index=movie_ratings['title_x'])
#Use get_recommendations to display results
get_recommendations('Iron Man', cosine_sim2)
