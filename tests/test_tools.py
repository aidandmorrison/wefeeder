import tools
import pytest

long_hastag_list = "'NetWorth'; 'sidehustle'; 'mumsthathustle'; 'sidehustlingmums'; 'moretime'; 'financialindependance'; " \
                "'financialfreedom'; 'mumsmakingmoney'; 'plantingourmoneytree'"

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
    case = long_hastag_list
    assert tools.hashtag_counter(case) == 9


def test_return_first_string_match():
    """Bare minimum test"""
    assert tools.return_first_string_match(long_hastag_list, ['savings', 'credit']) is None
    assert tools.return_first_string_match(long_hastag_list, ['hustle', 'financial']) == 'hustle'
