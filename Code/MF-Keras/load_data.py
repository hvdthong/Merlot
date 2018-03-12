import pandas as pd


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


if __name__ == "__main__":
    path_file_ = "../data/SO_GH/GHSO_users.csv"
    users = data_category(path_file=path_file_, name="GHSO_users")
    print users.head(), len(users)

    path_file_ = "../data/SO_GH/user_repository.csv"
    user_repository = data_category(path_file=path_file_, name="user_repository")
    print user_repository.head(), len(user_repository)

    path_file_ = "../data/SO_GH/user_question.csv"
    user_question = data_category(path_file=path_file_, name="user_question")
    print user_question.head(), len(user_question)
    print len(user_question.stackoverflow_question_id.unique())
    print len(user_question.stackoverflow_user_id.unique())
    exit()

    GH_SO_repository = user_repository.loc[
        user_repository['github_user_id'].isin(users['github_user_id'])]
    print GH_SO_repository.head(), len(GH_SO_repository)

    SO_GH_question = user_question.loc[
        user_question['stackoverflow_user_id'].isin(users['stackoverflow_user_id'])]
    print SO_GH_question.head(), len(SO_GH_question)

    users_SO_GH = users.loc[(users["stackoverflow_user_id"].isin(SO_GH_question["stackoverflow_user_id"])) &
                            (users["github_user_id"].isin(GH_SO_repository["github_user_id"]))]
    print users_SO_GH.head(), len(users_SO_GH)
