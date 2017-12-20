import pandas as pd
import numpy as np

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('../data/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']
movies = pd.read_csv('../data/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)
print("StarWars Similiar Movies")
#Agora a funcao pivot_table constrou um DataFrame com uma matriz user/movie rating.
#Percea que NaN indica que falta dados não classificados pelo usuario
movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
starWarsRatings = movieRatings['Star Wars (1977)']
similarMovies = movieRatings.corrwith(starWarsRatings)
similarMovies = similarMovies.dropna()
df = pd.DataFrame(similarMovies)
movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
#Listar ao menos titulos que foram avaliados por 100 pessoas 
popularMovies = movieStats['rating']['size'] >= 100
movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:15]
df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))
#Ordennar esses resultados pelo score de similaridade. Então o resultado é bem melhor :)
print(df.sort_values(['similarity'], ascending=False)[:15])