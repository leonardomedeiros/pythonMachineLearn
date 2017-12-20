import pandas as pd
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
#Pandas' corrwith function makes it really easy to compute the pairwise correlation of Star Wars' 
#vector of user rating with every other movie! After that,
# we'll drop any results that have no data, and construct a new DataFrame of movies 
#and their correlation score (similarity) to Star Wars:
similarMovies = movieRatings.corrwith(starWarsRatings)
similarMovies = similarMovies.dropna()
df = pd.DataFrame(similarMovies)
print(similarMovies.sort_values(ascending=False))
