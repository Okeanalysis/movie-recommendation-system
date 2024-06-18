# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = FastAPI()

# Load the data and model
with open('movie.pkl', 'rb') as file:
    data = pickle.load(file)

new_df = data['data']

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tag_matrix = tfidf_vectorizer.fit_transform(new_df['tag'])

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set templates directory
templates = Jinja2Templates(directory="templates")

def recommend(movie_title: str, new_df, tag_matrix, tfidf_vectorizer) -> List[str]:
    if movie_title in new_df['title'].values:
        # Get the index of the movie that matches the title
        movie_index = new_df[new_df['title'] == movie_title].index[0]
        
        # Calculate cosine similarities for the movie with all other movies
        similarities = cosine_similarity(tag_matrix[movie_index], tag_matrix).flatten()
        
        # Get indices of movies sorted by similarity score (excluding the movie itself)
        similar_indices = similarities.argsort()[::-1][1:11]
        
        # Get the titles of the top 10 most similar movies
        recommended_movies = new_df.iloc[similar_indices]['title'].tolist()
    else:
        # If the movie is not found, calculate similarity for the given title as a new entry
        query_vector = tfidf_vectorizer.transform([movie_title])
        similarities = cosine_similarity(query_vector, tag_matrix).flatten()
        
        # Get indices of movies sorted by similarity score
        similar_indices = similarities.argsort()[::-1][:10]
        
        # Get the titles of the top 10 most similar movies
        recommended_movies = new_df.iloc[similar_indices]['title'].tolist()
    
    return recommended_movies

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recommend", response_class=HTMLResponse)
async def recommend_name(request: Request, query: str = Form(...)):
    recommended_movies = recommend(query, new_df, tag_matrix, tfidf_vectorizer)
    return templates.TemplateResponse("index.html", {"request": request, "recommendations": recommended_movies})


