import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
import keras
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.utils import plot_model

dataset = pd.read_csv("../data/ml-100k/u.data", sep='\t', names="user_id,item_id,rating,timestamp".split(","))
dataset.user_id = dataset.user_id.astype('category').cat.codes.values
dataset.item_id = dataset.item_id.astype('category').cat.codes.values
train, test = train_test_split(dataset, test_size=0.2)
n_users, n_movies = len(dataset.user_id.unique()), len(dataset.item_id.unique())
n_latent_factors = 3
print n_users, n_movies

movie_input = keras.layers.Input(shape=[1], name='Item')
movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors, name='Movie-Embedding')(movie_input)
movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
user_input = keras.layers.Input(shape=[1], name='User')
user_vec = keras.layers.Flatten(name='FlattenUsers')(
    keras.layers.Embedding(n_users + 1, n_latent_factors, name='User-Embedding')(user_input))

prod = keras.layers.dot([movie_vec, user_vec], axes=1)
model = keras.models.Model([user_input, movie_input], prod)
model.compile('adam', 'mean_squared_error')
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)
model.summary()
