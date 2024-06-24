import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


df = pd.read_csv('movies_updated.csv')

mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df['genres'])

feature_matrix = pd.DataFrame(genres_encoded, columns=mlb.classes_)

cosine_sim = cosine_similarity(feature_matrix)

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

st.title('Movie Recommendation System')
st.write("dhruv says hi")

movie_title = st.selectbox(
    label ="select your movie",
    options = list(df["title"])
)

if movie_title:
    recommendations = get_recommendations(movie_title)
    st.write('Recommended Movies:')
    for i, movie in enumerate(recommendations, 1):
        st.write(f"{i}. {movie}")
