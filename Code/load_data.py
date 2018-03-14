import pandas as pd
import numpy as np


def data_category(path_file, name):
    if name == "GHSO_users":
        dataset = pd.read_csv(path_file, sep=',', names="github_user_id,stackoverflow_user_id".split(","))
        return dataset
    elif name == "user_question":
        dataset = pd.read_csv(path_file, sep=',', names="stackoverflow_user_id,stackoverflow_question_id".split(","))
        return dataset
    elif name == "user_repository":
        dataset = pd.read_csv(path_file, sep=',', names="github_user_id,github_repository_id".split(","))
        return dataset
    else:
        print "Unknown data category"
        exit()


def sample_data(df, name, size):
    if name == "user_question":
        users = df.stackoverflow_user_id.unique()
        items = df.stackoverflow_question_id.unique()
    elif name == "user_repository":
        users = df.github_user_id.unique()
        items = df.github_repository_id.unique()

    df["Relevant"] = 1
    negative_sample = list()
    for i in xrange(size):
        user = users[np.random.choice(users.shape[0], 1, replace=False)][0]
        item = items[np.random.choice(items.shape[0], 1, replace=False)][0]
        index = list()
        if name == "user_question":
            if df.loc[(df['stackoverflow_user_id'] == user) & (df["stackoverflow_question_id"] == item)].empty:
                index.append(user)
                index.append(item)
                negative_sample.append(np.array(index))
        elif name == "user_repository":
            if df.loc[(df['github_user_id'] == user) & (df["github_repository_id"] == item)].empty:
                index.append(user)
                index.append(item)
                negative_sample.append(np.array(index))
        print i
    negative_sample = np.array(negative_sample)
    if name == "user_question":
        print negative_sample.shape
        negative_sample = pd.DataFrame(negative_sample,
                                       columns=["stackoverflow_user_id", "stackoverflow_question_id"])
    elif name == "user_repository":
        negative_sample = pd.DataFrame(negative_sample,
                                       columns=["github_user_id", "github_repository_id"])
    negative_sample["Relevant"] = 0
    df = pd.concat([df, negative_sample])
    df = df.sample(frac=1.0)
    df.to_csv("./data/SO_GH/" + name + "_sample.csv", sep=',', index=False, header=False)


if __name__ == "__main__":
    # path_file_ = "./data/SO_GH/GHSO_users.csv"
    # users = data_category(path_file=path_file_, name="GHSO_users")
    # print users.head(), len(users)

    path_file_ = "./data/SO_GH/user_repository.csv"
    user_repository = data_category(path_file=path_file_, name="user_repository")
    sample_data(df=user_repository, name="user_repository", size=int(len(user_repository) * 0.9))
    # print user_repository.head(), len(user_repository)
    # print len(user_repository.github_user_id.unique()), len(user_repository.github_repository_id.unique())

    # path_file_ = "./data/SO_GH/user_question.csv"
    # user_question = data_category(path_file=path_file_, name="user_question")
    # sample_data(df=user_question, name="user_question", size=len(user_question))
    # print user_question.head(), len(user_question)
    # print len(user_question.stackoverflow_user_id.unique()), len(user_question.stackoverflow_question_id.unique())

    # GH_SO_repository = user_repository.loc[
    #     user_repository['github_user_id'].isin(users['github_user_id'])]
    # print GH_SO_repository.head(), len(GH_SO_repository)
    #
    # SO_GH_question = user_question.loc[
    #     user_question['stackoverflow_user_id'].isin(users['stackoverflow_user_id'])]
    # print SO_GH_question.head(), len(SO_GH_question)
    #
    # users_SO_GH = users.loc[(users["stackoverflow_user_id"].isin(SO_GH_question["stackoverflow_user_id"])) &
    #                         (users["github_user_id"].isin(GH_SO_repository["github_user_id"]))]
    # print users_SO_GH.head(), len(users_SO_GH)
