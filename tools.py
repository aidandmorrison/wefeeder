

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


