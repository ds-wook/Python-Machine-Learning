# %%
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings("ignore")

movies = pd.read_csv("./data/tmdb_5000_movies.csv")
print(movies.shape)
movies.head(1)

# %%
movies_df = movies[["id", "title", "genres", "vote_average", "vote_count","popularity", "keywords", "overview"]]
movies_df.head()

# %%
pd.set_option("max_colwidth", 100)
movies[["genres", "keywords"]][:1]

# %%
movies_df.info()
# %%
movies_df["genres"] = movies_df["genres"].apply(lambda x : eval(x))
movies_df["genres"] = movies_df["genres"].apply(lambda x : [dic["name"] for dic in x])
# %%
movies_df["keywords"] = movies_df["keywords"].apply(lambda x : eval(x))
movies_df["keywords"] = movies_df["keywords"].apply(lambda x : [dic["name"] for dic in x])

# %%
movies_df.head()

# %%
movies_df["genres_literal"] = movies_df["genres"].apply(lambda x : " ".join(x))
movies_df["genres_literal"].head()

# %%
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(min_df = 0, ngram_range=(1, 2))
genre_mat = count_vect.fit_transform(movies_df["genres_literal"])
print(genre_mat.shape)

# %%
from sklearn.metrics.pairwise import cosine_similarity

genre_sim = cosine_similarity(genre_mat, genre_mat)
print(genre_sim.shape)
print(genre_sim[:2])

# %%
genre_sim_index = genre_sim.argsort()[:, ::-1]
print(genre_sim_index[:2])

# %%
def find_sim_movie(df, sorted_ind, title_name, top_n = 10):
    title_movie = df[df["title"] == title_name]
    title_index = title_movie.index.values
    similar_indexes = sorted_ind[title_index, : top_n]
    similar_indexes = similar_indexes.flatten()
    return df.iloc[similar_indexes]

# %%
C = movies_df["vote_average"].mean()
m = movies_df["vote_count"].quantile(0.6)
print("C : {0:.3f}, m : {1:.3f}".format(C, m))

# %%
percentile = 0.6
m = movies_df["vote_count"].quantile(percentile)
C = movies_df["vote_average"].mean()

def weighted_vote_average(record):
    v = record["vote_count"]
    R = record["vote_average"]

    return (v / (v + m) * R) + (m / (m + v) * C)

# %%
def find_sim_movie(df, sorted_ind, title_name, top_n = 10):
    title_movie = df[df["title"] == title_name]
    title_index = title_movie.index.values

    similar_indexes = sorted_ind[title_index, : top_n * 2]
    similar_indexes = similar_indexes.flatten()

    similar_indexes = similar_indexes[similar_indexes != title_index]

    return df.iloc[similar_indexes].sort_values("weighted_vote", ascending = False)[:top_n]
