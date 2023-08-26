import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load the DataFrame
data = pd.read_csv("imdb_top_1000.csv")  # Replace with the actual file path

# Preprocess the data if needed
# ...

# Create a CountVectorizer to convert text data into a matrix of token counts
cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(data['Overview'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix)

# Set Streamlit page configuration
st.set_page_config(page_title='Movie Recommender App', layout='wide')

# Streamlit app
def main():
    st.title("Movie Recommender System")

    # Dropdown to select a movie
    selected_movie = st.selectbox("Select a movie", data['Series_Title'])

    # Find the index of the selected movie
    idx = data[data['Series_Title'] == selected_movie].index[0]

    # Get the cosine similarity scores for the selected movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get movie recommendations
    recommendations = sim_scores[1:6]  # Exclude the selected movie itself

    # Display the selected movie
    st.subheader("Selected Movie")
    st.write(selected_movie)
    selected_movie_genres = data.loc[idx, 'Genre']
    st.write(f"Genres: {selected_movie_genres}")

    # Display recommended movies in a table
    st.subheader("Top 5 Recommended Movies")
    recommended_movies = []
    for i, (movie_idx, score) in enumerate(recommendations):
        recommended_movie = data.loc[movie_idx, 'Series_Title']
        recommended_movie_genres = data.loc[movie_idx, 'Genre']
        recommended_movies.append((recommended_movie, recommended_movie_genres))
    
    rec_df = pd.DataFrame(recommended_movies, columns=['Recommended Movie', 'Genres'])
    st.table(rec_df)

# Run the app
if __name__ == '__main__':
    main()
