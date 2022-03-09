# wefeeder
Small example method on how one could rank posts for user feeds

## Instructions to intall and run

I've used a conda virtual env named `wefeeder_env`, you'll need conda installed
Clone into the repo, then:
```
conda env create -f environment.yml
conda activate wefeeder_env
python -m textblob.download_corpora
python demo.py
```

## Thinking about the approach and process

How should we best rank the posts for each user?

We could divide the possible approaches into two main categories:

1.	Learned relationships
2.	Assumed relationships

Generally, data scientists would (and should) strongly prefer to use learned relationships.  I.e. there’s something in the data that shows us degrees of interest for some observations, and we build a model based on those observations to infer/predict what their degree of interest will be for new (unseen) observations.  This is basically trying to structure the problem for supervised machine learning. 

Otherwise, we can use assumed relationship, as suggested in the instructions.   Suggestions 1-3 are all direct assumptions (potentially sensible ones).  Suggestion 4 relies on the premise that people pay attention to users who are similar or hold similar interests.  Again, it seems a sensible assumption, but it remains an assumption unless there’s actually evidence in the data to support it.  Another possible additional assumed relationship which could be plausible is that people are interested in posts that look similar to their own (or their most recent).  We could use bag-of-words or any more sophisticated NLP methods (including sentiment) to try to give a distance metric between each post. Again, how you define the distance metrics for this is all up for grabs, and quite assumption reliant.  Essentially the adoption of those distance metrics make it an unsupervised machine learning problem. 

As such my first instinct is to look for ways of structuring this as a supervised problem, and see what can be learned from the data. 

The first really big issue here is that no data is provided about how users engaged with or responded to other posts, other than the replies. And the replies are quite sparse, only 5 posts got any replies, and there were only 11 total replies. (At least they were replied to by users in the set.)

To try to think through what approaches are viable, I’ll briefly describe the likely minimum data we’d need for two approaches of machine-learning:

## Learned Relationships

### Per-user model
We train a unique model for every user.  We would need a decent number of scored ‘observations’ for every user.  An observation would be a user’s interaction with a post, which we could use to deduce an ‘interest score’ with some assumptions (view time, vs likes, comments etc).  A decent number of observations would be a number at least 2 or 3x the number of features we might use in the model, which could be dozens or even more, and at very least we’d want third/quarter to be non-trivial ( > ~7) so that we can validate somewhat using cross-validation .  We wouldn’t care about the actual attributes of the user themselves, (i.e. their age) because training a unique model for every person, so any consequence of that attribute could and should be learned from the data.  We could use attributes of the users making posts as features (age of poster).

Obviously we don’t have anything close to the right amount of data to do this.

### Multi-user model
The obvious alternative is to train a single model for a group of users. In reality you might segment your users using some unsupervised clustering (eg new users, active users, older users, by gender, net worth etc), but here let’s consider just lumping all users to gether into a single group, and trying to train a model on the whole population. 

All the same rough rules apply about data requirements apply, but in this case since we’re incorporating features that relate to each post, and to both the poster, and post-viewer, and probably relative metrics between the two (how much more/less old they are, wealthy, or indebted they are) as well as state at time of viewing (or posting?) we’re naturally going to have a much larger feature space, and will consequently need a much larger number of observations to train a model that might hope to learn relationships between any/all of them. 

In this case we only have 5 posts that have any replies at all, and 11 users who made replies. So it would be really quite impossible to train this kind of model with any hope of success with this data, and certainly wouldn’t be possible to cross-validate it.  

The advantage of this approach relative to the per-user model is that we wouldn’t need a minimum number of observations for every user. In essence, this model would assume that the users with similar traits and characteristics have similar preferences.  It would not be dissimilar to using some kind of nearest-neighbour assumption to transfer a model from users we have observations on, to users we don’t, except that the choice of characteristics on which we determine the distance metrics would be less arbitrary, and actually learned from the data. I.e. if age doesn’t usefully distinguish between users reactions to a given post, then age won’t be used (or not highly weighted) in considering how close/similar a user would be in response to a post.

### Universal user model
Or, another really rash alternative would be just to assume that all users essentially have the same mind, and have basically the same reaction to all posts.  We could train a model that just ranks the posts in terms of a universal merit score, and bump those posts up everyone’s feeds. Of course that’s miles from reality. However, there is probably some slight thread of merit, particularly in a really sparse data environment like we have in this exercise.  To the extent that there are a significant number of rubbish/useless/boring/spam posts that almost no-one will be interested in, this kind of model could be useful in weeding those out.  (At least, it would do if we didn’t have 90% of the sample all with degenerate ranking of no observable interest.) But the opposite case, where we have some excellent/interesting posts which are likely to be interesting to basically everyone, could well be consistent with this data (a few posts get multiple replies), and a worthy part of a ranking system.

Here we have 50 observations, with quite a skewed distribution (only 5 get a reply).  But since we can discard all information about the post viewer (and only include information about the post itself, and post-writer) there’s a chance that 50 observations captures something. It might be reasonable given the circumstances just treat this as a binary prediction (gets reply, or not).  In which case we still don’t have a fraction of as much data as we’d like, but there’s a chance that a very strong signal could be found, and more importantly, there’s also a slight chance that we might be get a tiny hint as to whether any perceived signal is significant/coincidental using a ROC AUC score.  We’d only have couple of possible train-test splits with positive results in both parts of the data, but it might just give us the right kind of hint. 

## Assumed Relationships
The assumptions 1-4 in the problem statement all seem valid, and straight-forward enough. I actually think that perhaps a very plausible assumption to work with wasn’t listed: that people are interested in similar posts to what they themselves post about. 

### Similar Posts Assumption
I think that this would be a good one to work with for a couple of reasons:
-	Intuitively it seems likely enough.  Even if you also want to discover answers/inspiration that doesn’t look like similar text to your posts (which might be more questions/complaints perhaps) it’s probably still going to be nice to see that other people are thinking/saying similar things to you.  Including if you’re feeling a bit lost etc. Or if you’re posting advice, seeing your ‘finfluencer’ competitors posting their take on similar topics is probably also interesting. 
-	This is by far the richest piece of data that we have populated for all users. A median string of 140 characters is pretty good is a lot more information than a date of birth, or a couple of interests.
-	There are potentially quite computationally efficient ways of implementing this at runtime, if we take something like a bag-of-words approach. We just need a document-term-matrix for each user’s posts (or recent posts, or in this case only post), and that’s essentially the stored model for each user. Each post (arriving in live production) will also need to have its own DTM generated once.  The distance metric is very simple and mathematical, and could be implemented in a low-level language if needed for fast execution. 
-	Even with this much data, there’s a good chance of getting a decent number of non-degenerate rankings.  I.e. I would guess/hope that for lots of the posts, at least a couple of relevant words would also be present in more than a couple other posts.  Of course, the definition of what words are relevant could take quite a bit of tuning, but using the whole corpus here to inform this would be a good start, alongside other trivial default methods. 

## Conclusion about approach

•	We’ll try to train a universal model on a binary ‘has replies’ label with only post and poster data.  We should stick to ~5 features, so that we don’t wind up with a grossly over-fit model, but maybe we’ll just have to stretch that a bit. 
•	We’ll stick to simple models, probably start with a decision tree (little point trying more complex ones with so little data).
•	If we get a model that has any hint of a signal, we’ll implement that as a first pass for all users.  This result may well leave a lot of degenerate predictions with only a couple of levels separate from a simple tree.
•	For any/all posts that remain degenerate, we’ll use the ‘similar posts’ approach based on a bag of words.
•	For any/all reposts that remain degenerate after that we’ll revert to the post_time field.  We could have used any other of the assumptions including in combinations, I’m reverting to simplest/easiest because I think the above two methods will be relatively involved.
