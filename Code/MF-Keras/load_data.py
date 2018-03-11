import pandas as pd


def data_category(path_file, name):
    if name == "SOGH_users":
        dataset = pd.read_csv(path_file, sep=',', names="stackoverflow_user_id,github_user_id".split(","))
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
    path_file_ = "../data/SO_GH/SOGH_users.csv"
    users = data_category(path_file=path_file_, name="SOGH_users")
    print users.head()

    path_file_ = "../data/SO_GH/user_question.csv"
    user_question = data_category(path_file=path_file_, name="user_question")
    print user_question.head()
    print len(user_question)

    print type(users), type(user_question)
