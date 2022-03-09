import tools
import pytest
import pandas as pd

long_hastag_list = "'NetWorth'; 'sidehustle'; 'mumsthathustle'; 'sidehustlingmums'; 'moretime'; 'financialindependance'; " \
                "'financialfreedom'; 'mumsmakingmoney'; 'plantingourmoneytree'"

@pytest.fixture
def fitted_mod():
    mod = tools.PostFeeder("../")
    mod.load_data()
    mod.fit()
    return mod

def test_hashtag_counter_non_string():
    """Check that we get the appropriate error message """
    cases = [1, 0, None, False, True]
    for case in cases:
        with pytest.raises(ValueError):
            tools.hashtag_counter(case)


def test_hashtag_counter_empty():
    """Just want to see if the obvious empty examples get the right result"""
    cases = ["", "[]", "0"]
    for case in cases:
        assert tools.hashtag_counter(case) == 0


def test_hashtag_counter_long():
    """Take an example from the data. We should try to generalise further for more vigorous testing, but it's a start"""
    case = long_hastag_list
    assert tools.hashtag_counter(case) == 9


def test_return_first_string_match():
    """Bare minimum test"""
    assert tools.return_first_string_match(long_hastag_list, ['savings', 'credit']) is None
    assert tools.return_first_string_match(long_hastag_list, ['hustle', 'financial']) == 'hustle'


def test_data_loads():
    mod = tools.PostFeeder("../")
    mod.load_data()
    assert isinstance(mod.posts, pd.DataFrame)
    assert mod.posts.shape[0] > 0
    assert isinstance(mod.users, pd.DataFrame)
    assert mod.users.shape[0] > 0


def test_model_fits():
    mod = tools.PostFeeder("../")
    mod.load_data()
    mod.fit()
    assert isinstance(mod.tfidf, pd.DataFrame)
    assert mod.tfidf.shape[0] == mod.posts.shape[0]


def test_model_predicts(fitted_mod):
    """
    This is just a minimal test, somewhat bespoke to this data, and hence not ideal.
    We'll just take two examples, and check that the order changes from original, and that they're not both the same.
    (the latter part risks some flakiness if we generalised this to any sample of users)
    """

    # Prep model
    # mod = tools.PostFeeder("../")
    # mod.load_data()
    # mod.fit()
    mod = fitted_mod

    # Prepare relevant objects for each user, checking that we've removed one entry to derive a 'null' ranking for each
    user0_id = mod.users['uid'].iloc[0]
    local_posts = mod.posts.copy()
    user0_post_id = local_posts[local_posts['uid'] == user0_id]['post_id']
    user0_post_id = user0_post_id[user0_post_id.index[0]]
    local_posts = mod.posts.copy()
    user0_null_rank = local_posts[local_posts['uid'] != user0_id]['post_id'].to_list()
    assert len(user0_null_rank) == mod.posts.shape[0] - 1
    user1_id = mod.users['uid'].iloc[1]
    local_posts = mod.posts.copy()
    user1_post_id = local_posts[local_posts['uid'] == user1_id]['post_id']
    user1_post_id = user1_post_id[user1_post_id.index[0]]
    local_posts = mod.posts.copy()
    user1_null_rank = local_posts[local_posts['uid'] != user1_id]['post_id'].to_list()
    assert len(user1_null_rank) == mod.posts.shape[0] - 1

    # Predict, and check we're not predicting our own post
    user0_pred_rank = mod.predict(user0_id)
    assert user0_id not in user0_pred_rank
    user1_pred_rank = mod.predict(user1_id)
    assert user1_id not in user1_pred_rank

    # Checking that the rankings have produced new, distinguished rankings
    assert user0_pred_rank != user0_null_rank
    assert user1_pred_rank != user1_null_rank
    user1_pred_rank.remove(user0_post_id)
    user0_pred_rank.remove(user1_post_id)
    assert user1_pred_rank != user0_pred_rank


