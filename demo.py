import pandas as pd
from tools import PostFeeder

if __name__ == "__main__":
    users = pd.read_csv("users.csv")
    all_uids = users['uid'].to_list()
    mod = PostFeeder()
    mod.load_data()
    mod.fit()

    for uid in all_uids:
        print("User: " + uid + " should get posts in sequence: ")
        print(mod.predict(uid))