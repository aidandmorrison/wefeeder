import pandas as pd

import tools
from tools import hashtag_counter, return_first_string_match
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word

#%%
interests = pd.read_csv("interests.csv")
posts = pd.read_csv("posts.csv")
users = pd.read_csv("users.csv")

#%%
# Create the label for posts 'has_reply'
parent_id_list = list(posts['parent_id'].unique())
posts['has_reply'] = posts['post_id'].apply(lambda x: x in parent_id_list)

#%%
# Let's join together the data and one-hot-encode the interests
X = posts.copy()
X = X.merge(users, on="uid")
interests['has_interest'] = [True for i in range(interests.shape[0])]
interests_wide = interests.pivot_table(index = ["uid"], columns='interest', values='has_interest').fillna(0)
X = X.merge(interests_wide, on="uid")

#%%
# Let's get the hashtags sorted out so they can be one-hot-encoded too
tags = posts.loc[:, ['post_id', 'hashtags']]
tags['hashtags'] = tags['hashtags'].apply(lambda x: x.lower())
print(tags.hashtags.unique())
# This actually won't be worth one-hot encoding, too many unique values.
# Let's just count the number of hashtags as a feature, and flag hashtags that match an interest
tags['tag_count'] = tags['hashtags'].apply(hashtag_counter)

#%%
# Finding hashtags contain words that are also interests
interest_words = []
for i in interests['interest'].unique():
    words = i.split(' ')
    interest_words.extend([w.lower() for w in words])

# make assumption that we'll generally get one (at best) full match with an interest in all hashtags
tags['interest_match'] = tags['hashtags'].apply(return_first_string_match, args=(interest_words,))
tags['interest_match'].fillna("None", inplace = True)
tags['has_match'] = [True for i in range(tags.shape[0])]
tags_wide = tags.pivot_table(index= 'post_id', columns='interest_match', values='has_match')
tags_wide.reset_index(inplace=True)
tags_wide.fillna(0, inplace=True)
tags = tags.merge(tags_wide, on="post_id")

#%%
# Do final parsing, and selection of relevant variables
X = X.merge(tags.drop('hashtags', axis=1), on="post_id")
X['dob'] = pd.to_datetime(X['dob'], dayfirst=True).astype(int)/10**9
X['post_time'] = pd.to_datetime(X['post_time']).astype(int)/10**9
y = X['has_reply']
drop_list = ['first_name','family_name', 'parent_id', 'uid', 'text', 'hashtags', 'post_id',
             'parent_id', 'interest_match', 'None', 'a', 'has_reply']
X.drop(drop_list, axis = 1, inplace=True)

#%%
# Train a model
for seed in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed, shuffle=True)
    mod = DecisionTreeClassifier()
    mod.fit(X_train, y_train)
    plt.show()
    preds = mod.predict_proba(X_test)
    try:
        score = roc_auc_score(y_test, preds)
        print(score)
    except ValueError:
        print("no positive cases in test sample, or all the same prediction")

# This is clearly not viable with such a skewed dataset, and so few observations relative to features

#%%
# Create dtf (actually a tf-idf) based on the posts
vectorizer = TfidfVectorizer(stop_words='english', max_features=50, min_df=2, ngram_range=(1,2))
lemmatised_docs =posts['text'].apply(lambda x: Word(x).lemmatize())
doc_vec = vectorizer.fit_transform(lemmatised_docs)
tfidf = pd.DataFrame(doc_vec.toarray().transpose(), index = vectorizer.get_feature_names())
tfidf.columns = posts['post_id']
tfidf = tfidf.T

# A little experimentation with the max_features and min_df revealed that better differentiation...
# seems to occur with roughly this number of features and minimum frequency.
# The words and word pairs seem to make sense and seem appropriate

#%%
# Just test out one example

def get_square_dist(vec1, vec2):
    return sum((vec1 - vec2)**2)
example = tfidf.iloc[0,:]
comparison = tfidf.iloc[1,:]
distance = get_square_dist(example, comparison)


#%%
# Figure out how it looks on average, particularly how many equal distances there are
all_users = []
all_unique_scores = []
for uid in users['uid'].unique():
    all_scores = []
    user_post = posts[posts['uid'] == uid]
    post_id = user_post['post_id']
    post_id = post_id[post_id.index[0]]
    term_scores = tfidf.loc[post_id,:]
    for id in tfidf.index:
        if id != post_id:
            this_dist = get_square_dist(term_scores, tfidf.loc[id,:])
            all_scores.append({"post_id": id, "distance": this_dist})
    rank_frame = pd.DataFrame(all_scores)
    degenerate = posts.shape[0] - len(rank_frame['distance'].unique())
    unique_scores = len(rank_frame['distance'].unique())
    all_users.append(degenerate)
    all_unique_scores.append(unique_scores)

print(sum(all_users)/len(all_users))
print(sum(all_unique_scores)/len(all_unique_scores))

# Roughly half of the other posts distinguished on average. Not too bad an outcome.

#%%
mod = tools.PostFeeder()
mod.load_data()
mod.fit()
out = mod.predict('e31e7cf3-9c4b-4da7-be33-db0e0cf4edd7')






