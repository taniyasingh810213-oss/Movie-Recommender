import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Set page title and layout
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Select a movie you like, and our AI will recommend similar titles based on genres!")

# Sample dataset (You can expand this later with a CSV file)
data = {
    'movie_id': [1, 2, 3, 4, 5, 6, 7],
    'title': ['The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 'Toy Story', 'Finding Nemo', 'Avengers: Endgame'],
    'genres': [
        'Action Sci-Fi', 
        'Action Sci-Fi Thriller', 
        'Adventure Drama Sci-Fi', 
        'Action Crime Drama', 
        'Animation Comedy Family', 
        'Animation Comedy Adventure', 
        'Action Adventure Sci-Fi'
    ]
}

df = pd.DataFrame(data)

# Preprocess genres and compute similarity matrix (cached so it runs fast)
@st.cache_resource
def load_similarity_matrix():
    cv = CountVectorizer(tokenizer=lambda x: x.split(), lowercase=False)
    count_matrix = cv.fit_transform(df['genres'])
    return cosine_similarity(count_matrix, count_matrix)

cosine_sim = load_similarity_matrix()

# Recommendation function
def get_recommendations(title):
    if title not in df['title'].values:
        return []
        
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4] # Get top 3 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# --- UI Components ---
selected_movie = st.selectbox(
    "Choose a movie from the database:",
    df['title'].values
)

if st.button("Get Recommendations", type="primary"):
    with st.spinner("Finding similar movies..."):
        recommendations = get_recommendations(selected_movie)
        
    st.subheader(f"Because you liked **{selected_movie}**, you might also enjoy:")
    for i, rec in enumerate(recommendations, 1):
        st.success(f"{i}. {rec}")
        