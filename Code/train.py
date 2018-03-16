import pandas as pd
from sklearn.model_selection import train_test_split
from MF_nn import MF_nonnegative
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from keras.utils import plot_model


def loading_GHSO_data(path_file, name):
    if name == "user_question":
        dataset = pd.read_csv(path_file, sep=',',
                              names="stackoverflow_user_id,stackoverflow_question_id,relevant".split(","))
        dataset.stackoverflow_user_id = dataset.user_id.astype('category').cat.codes.values
        dataset.stackoverflow_question_id = dataset.stackoverflow_question_id.astype('category').cat.codes.values

        return dataset
    elif name == "user_repository":
        dataset = pd.read_csv(path_file, sep=',', names="github_user_id,github_repository_id,relevant".split(","))
        return dataset
    else:
        print "Unknown data category"
        exit()


if __name__ == "__main__":
    # dataset = pd.read_csv("./data/ml-100k/u.data", sep='\t', names="user_id,item_id,rating,timestamp".split(","))
    # dataset.user_id = dataset.user_id.astype('category').cat.codes.values
    # dataset.item_id = dataset.item_id.astype('category').cat.codes.values
    # train, test = train_test_split(dataset, test_size=0.2)
    # n_users, n_movies = len(dataset.user_id.unique()), len(dataset.item_id.unique())
    # n_latent_factors_user = 5
    # n_latent_factors_movie = 8
    # print n_users, n_movies
    #
    # mf_nn = MF_nonnegative(n_user=n_users, n_item=n_movies, n_hidden=64, n_latent_ftr=8)
    # mf_nn.build_graph(model="user_item")
    # mf_nn.model.summary()
    # history = mf_nn.model.fit([train.user_id, train.item_id], train.rating, epochs=20, verbose=0)
    # y_hat_2 = np.round(mf_nn.model.predict([test.user_id, test.item_id]), 0)
    # y_true = test.rating
    # print(mean_absolute_error(y_true, y_hat_2))
    # print(mean_absolute_error(y_true, mf_nn.model.predict([test.user_id, test.item_id])))

    # dataset = pd.read_csv("./data/SO_GH/user_question_sample.csv", sep=',',
    #                       names="stackoverflow_user_id,stackoverflow_question_id,relevant".split(","))
    # dataset.stackoverflow_user_id = dataset.stackoverflow_user_id.astype('category').cat.codes.values
    # dataset.stackoverflow_question_id = dataset.stackoverflow_question_id.astype('category').cat.codes.values
    # train, test = train_test_split(dataset, test_size=0.1)
    # n_users, n_questions = len(dataset.stackoverflow_user_id.unique()), len(dataset.stackoverflow_question_id.unique())
    #
    # mf_nn = MF_nonnegative(n_user=n_users, n_item=n_questions, n_hidden=64, n_latent_ftr=128)
    # mf_nn.build_graph(model="user_item")
    # mf_nn.model.summary()
    # # plot_model(mf_nn.model, to_file="model_user_question.png", show_shapes=True, show_layer_names=True)
    # # mf_nn.model.summary()
    # # exit()
    # history = mf_nn.model.fit([train.stackoverflow_user_id, train.stackoverflow_question_id], train.relevant, epochs=1,
    #                           verbose=1)
    # y_pred = np.round(mf_nn.model.predict([test.stackoverflow_user_id, test.stackoverflow_question_id]), 0)
    # y_true = test.relevant
    # print "accuracy: ", accuracy_score(y_true=y_true, y_pred=y_pred)
    # print "precision: ", precision_score(y_true=y_true, y_pred=y_pred)
    # print "recall: ", recall_score(y_true=y_true, y_pred=y_pred)
    # print "F1: ", f1_score(y_true=y_true, y_pred=y_pred)

    dataset = pd.read_csv("./data/SO_GH/user_repository_sample.csv", sep=',',
                          names="github_user_id,github_repository_id,relevant".split(","))
    dataset.github_user_id = dataset.github_user_id.astype('category').cat.codes.values
    dataset.github_repository_id = dataset.github_repository_id.astype('category').cat.codes.values
    train, test = train_test_split(dataset, test_size=0.1)
    n_users, n_questions = len(dataset.github_user_id.unique()), len(dataset.github_repository_id.unique())

    mf_nn = MF_nonnegative(n_user=n_users, n_item=n_questions, n_hidden=64, n_latent_ftr=128)
    mf_nn.build_graph(model="user_item")
    mf_nn.model.summary()
    # plot_model(mf_nn.model, to_file="model_user_repository.png", show_shapes=True, show_layer_names=True)
    # mf_nn.model.summary()
    # exit()
    history = mf_nn.model.fit([train.github_user_id, train.github_repository_id], train.relevant, epochs=1,
                              verbose=1)
    y_pred = np.round(mf_nn.model.predict([test.github_user_id, test.github_repository_id]), 0)
    y_true = test.relevant
    print "accuracy: ", accuracy_score(y_true=y_true, y_pred=y_pred)
    print "precision: ", precision_score(y_true=y_true, y_pred=y_pred)
    print "recall: ", recall_score(y_true=y_true, y_pred=y_pred)
    print "F1: ", f1_score(y_true=y_true, y_pred=y_pred)
