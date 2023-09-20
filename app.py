import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_distances
import nltk

nltk.download('stopwords')
nltk.download('punkt') 

if 'model' not in st.session_state or 'data' not in st.session_state or 'matrix' not in st.session_state:
    model = pickle.load(open('model/tfidf.pkl', 'rb'))
    loaded_sparse_matrix = pickle.load(open('model/similarity.pkl', 'rb'))
    st.session_state['model'] = model
    st.session_state['matrix'] = loaded_sparse_matrix
    df = pd.read_csv('dataset/imdb_top_meta.csv')
    st.session_state['data'] = df


st.title(':tv: Recommendation System')
search, image = st.columns([3,1], gap='large')

with search:
    movies = st.selectbox(
        'Pick movies you\'ve watched',
        st.session_state['data']['Series_Title'].tolist()
    )
with image:
    image_link = st.session_state['data'][st.session_state['data']['Series_Title'] == movies]['Poster_Link'].to_list()[0]
    title_text = movies
    st.image(image_link, width=70)

if st.button('Recommend Me Movies'):
    st.subheader('Movies you\'ll likes :heart:')
    idx = st.session_state['data'][st.session_state['data']['Series_Title'] == movies].index[0]
    content = st.session_state['data'].loc[idx, 'tag']
    watched = st.session_state['model'].transform([content])
    dist = cosine_distances(watched, st.session_state['matrix'])
    rec_idx = dist.argsort()[0, 1:11]
    col0, col1, col2, col3, col4 = st.columns(5)
    col5, col6, col7, col8, col9 = st.columns(5)
    list_col = [col0, col1, col2, col3, col4, col5, col6, col7, col8, col9]
    for idx, val in enumerate(list_col):
        link = st.session_state['data'].loc[rec_idx].iloc[idx]['Poster_Link']
        title = st.session_state['data'].loc[rec_idx].iloc[idx]['Series_Title']
        with val:
            st.image(link, caption=title, width=150)

    st.write('Dataset Detail')
    st.dataframe(st.session_state['data'].loc[rec_idx][['Series_Title', 'Genre', 'Director','tag']])
else:
    st.write('Waiting')