from keras.constraints import non_neg
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
from sklearn.metrics import mean_absolute_error

dataset = pd.read_csv("../data/ml-100k/u.data", sep='\t', names="user_id,item_id,rating,timestamp".split(","))
dataset.user_id = dataset.user_id.astype('category').cat.codes.values
dataset.item_id = dataset.item_id.astype('category').cat.codes.values
train, test = train_test_split(dataset, test_size=0.2)
n_users, n_movies = len(dataset.user_id.unique()), len(dataset.item_id.unique())
n_latent_factors_user = 5
n_latent_factors_movie = 8

movie_input = keras.layers.Input(shape=[1], name='Item')
movie_embedding = keras.layers.Embedding(n_movies + 1, n_latent_factors_movie, name='Movie-Embedding')(movie_input)
movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
movie_vec = keras.layers.Dropout(0.2)(movie_vec)

user_input = keras.layers.Input(shape=[1], name='User')
user_vec = keras.layers.Flatten(name='FlattenUsers')(
    keras.layers.Embedding(n_users + 1, n_latent_factors_user, name='User-Embedding')(user_input))
user_vec = keras.layers.Dropout(0.2)(user_vec)

concat = keras.layers.concatenate([movie_vec, user_vec])
concat_dropout = keras.layers.Dropout(0.2)(concat)
dense = keras.layers.Dense(200, name='FullyConnected')(concat)
dropout_1 = keras.layers.Dropout(0.2, name='Dropout')(dense)
dense_2 = keras.layers.Dense(100, name='FullyConnected-1')(concat)
dropout_2 = keras.layers.Dropout(0.2, name='Dropout')(dense_2)
dense_3 = keras.layers.Dense(50, name='FullyConnected-2')(dense_2)
dropout_3 = keras.layers.Dropout(0.2, name='Dropout')(dense_3)
dense_4 = keras.layers.Dense(20, name='FullyConnected-3', activation='relu')(dense_3)

result = keras.layers.Dense(1, activation='relu', name='Activation')(dense_4)
adam = Adam(lr=0.005)
model = keras.models.Model([user_input, movie_input], result)
model.compile(optimizer=adam, loss='mean_absolute_error')

plot_model(model, to_file="model_nn.png", show_shapes=True, show_layer_names=True)
model.summary()

history = model.fit([train.user_id, train.item_id], train.rating, epochs=50, verbose=0)
y_hat_2 = np.round(model.predict([test.user_id, test.item_id]), 0)
y_true = test.rating
print(mean_absolute_error(y_true, y_hat_2))
print(mean_absolute_error(y_true, model.predict([test.user_id, test.item_id])))
