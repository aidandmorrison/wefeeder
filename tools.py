import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word
import time

def hashtag_counter(hashtags: str) -> int:
    """
    Simple function to derive the number of hashtags detected in the raw input of the hashtags column
    of the posts data.  This is just derived from observation of the input data format, and relies upon
    there being one semicolon between each hashtag, but non after final one.
    Args:
        hashtags: string representing list of hashtags, separated by semicolon

    Returns: integer count of the number of hashtags

    """
    # Let's error explicitly if we don't get a string
    if not isinstance(hashtags, str):
        raise ValueError("Hashtags must comprise a string")

    # First detect empty cases.
    #  <3 is safe since empty values still have square braces, and any full ones would have ' at either end
    if len(hashtags) < 3:
        return 0

    # Then use the number of semicolons to determine count
    sc_count = hashtags.count(';')
    if sc_count > 0:
        return sc_count + 1
    else:
        return 1


def return_first_string_match(hashtags_long_string:str, matches: list[str]) -> str:
    """Simple function to check if there's an interest present in hastags, and if so return it"""
    for match in matches:
        if hashtags_long_string.__contains__(match):
            return match


def get_square_dist(vec1, vec2):
    return sum((vec1 - vec2)**2)


class PostFeeder:
    """
    This is the class object we'll use to implement the algorithm.
    The basic algorithm will first use a distance calculation to prefer posts that are
    similar to the user's own post.
    Then most recent times will be used.
    """

    def __init__(self, path_to_data: str = ""):
        self.path_to_data: str = path_to_data
        self.tfidf: [pd.DataFrame, None] = None
        self.posts: [pd.DataFrame, None] = None
        self.users: [pd.DataFrame, None] = None
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=50, min_df=2, ngram_range=(1,2))

    def load_data(self):
        self.posts = pd.read_csv(self.path_to_data + "posts.csv")
        self.posts['post_time'] = pd.to_datetime(self.posts['post_time']).astype(int)/10**9
        self.users = pd.read_csv(self.path_to_data + "users.csv")

    def fit(self):
        lemmatised_docs = self.posts['text'].apply(lambda x: Word(x).lemmatize())
        doc_vec = self.vectorizer.fit_transform(lemmatised_docs)
        tfidf = pd.DataFrame(doc_vec.toarray().transpose(), index=self.vectorizer.get_feature_names())
        tfidf.columns = self.posts['post_id']
        self.tfidf = tfidf.T

    def predict(self, uid: str, current_time: float = time.time()) -> list[str]:

        """
        This the prediction step to make a call for a specific user.
        This implementation relies on the convenient 1:1 relationship between users and posts in this
        data, which would need to be somewhat generalised in real production.
        Nothing here is well-optimised for performance.  All use of pandas should be removed (critical
        for memory overheads, and generally for performance), and python, or raw python could be removed too.

        Args:
            uid: string representing user
            current_time: seconds since epoch (1970)

        Returns: list of the strings representing post ids to present, in order

        """

        # First check we have the user
        if uid not in self.posts['uid'].to_list():
            raise ValueError("Can't find this user in our data")

        # Prepare objects for the selected user
        all_scores = []
        user_post = self.posts[self.posts['uid'] == uid]
        post_id = user_post['post_id']
        post_id = post_id[post_id.index[0]]
        post_time = user_post['post_time']
        post_time = post_time[post_time.index[0]]
        term_scores = self.tfidf.loc[post_id, :]

        # Walk through the
        for id in self.tfidf.index:
            if id != post_id:
                this_dist = get_square_dist(term_scores, self.tfidf.loc[id, :])
                all_scores.append({"post_id": id, "distance": this_dist, "post_time": post_time})
        rank_frame = pd.DataFrame(all_scores)

        # Finally for any posts that are still degenerate after this ranking, sort by most recent
        # The 'abs' below a bit over-egged, since we should assume that current time is after all times in training
        # The above observation actually makes the current_time redundant, as it will never change order of past posts
        rank_frame['time_diff'] = abs(current_time - rank_frame['post_time'])
        rank_frame.sort_values(by=['distance', 'time_diff'], inplace=True)

        # Return as a list of post ids
        return rank_frame['post_id'].to_list()

