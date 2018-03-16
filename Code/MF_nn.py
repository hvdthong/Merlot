import keras
from keras.optimizers import Adam
from keras.constraints import non_neg


class MF_nonnegative(object):
    def __init__(self, n_user, n_item, n_latent_ftr, n_hidden):
        self.n_user = n_user
        self.n_item = n_item
        self.n_latent_ftr = n_latent_ftr
        self.n_hidden = n_hidden

    def _create_embedding_item(self):
        self.item_input = keras.layers.Input(shape=[1], name='Item')
        self.item_embedding = keras.layers.Embedding(self.n_item + 1, self.n_latent_ftr,
                                                     name='Item-Embedding', embeddings_constraint=non_neg())(
            self.item_input)
        self.item_vec = keras.layers.Flatten(name='FlattenItems')(self.item_embedding)
        self.item_vec = keras.layers.Dropout(0.2)(self.item_vec)

    def _create_embedding_user(self):
        self.user_input = keras.layers.Input(shape=[1], name='User')
        self.user_vec = keras.layers.Flatten(name='FlattenUsers')(
            keras.layers.Embedding(self.n_user + 1, self.n_latent_ftr, name='User-Embedding',
                                   embeddings_constraint=non_neg())(self.user_input))
        self.user_vec = keras.layers.Dropout(0.2)(self.user_vec)

    def _create_concatenate_user_item(self):
        self.concat = keras.layers.concatenate([self.item_vec, self.user_vec])

    def _create_hidden_layer(self):
        dense = keras.layers.Dense(self.n_hidden, activation='relu', name='Activation')(self.concat)
        self.dense = keras.layers.Dropout(0.2, name='Dropout')(dense)

    def _create_output_layer(self):
        self.result = keras.layers.Dense(1, activation='sigmoid', name='output')(self.dense)
        self.adam = Adam(lr=0.005)

    def _create_model_user_item(self):
        self.model = keras.models.Model([self.user_input, self.item_input], self.result)
        self.model.compile(optimizer=self.adam, loss='binary_crossentropy', metrics=['accuracy'])

    def build_graph(self, model):
        if model == "user_item":
            self._create_embedding_item()
            self._create_embedding_user()
            self._create_concatenate_user_item()
            self._create_hidden_layer()
            self._create_output_layer()
            self._create_model_user_item()
        else:
            print "You need to input correct model name"
