## WIP: Trump Tweet Analysis

This project stems from two overarching questions:

#### Which emotions do politicians most frequently appeal to?

I recently saw a [BuzzFeed presentation](https://www.youtube.com/watch?v=4sc7vKo87qs) on, among other things, the virality of their content. A big part of their business relies on understanding what kind of content goes viral and why, so their data science team understandably spends a lot of time not only looking at how a piece of content becomes widely popular, but also looking at the distribution of content types in their most popular pieces of content. If you're familiar with BuzzFeed, the results are probably  in line with your expectations. Happiness and humor account for a very large proportion of viral posts. So do posts that shock, posts that try to draw an association between a part of your individual personality and a larger community, etc. 

This got me thinking about the distribution of content type for political content. Happiness and humor, which is the most popular content type for a company like BuzzFeed, is pretty much excluded in the political domain. Jeb Bush or Hillary Clinton can't post cat videos or crack jokes on a regular basis; they have to operate within a narrower specturm of messages: serious, fearful, angry, hopeful , charismatic, etc.

The question is this: what is the distribution of emotions for political content? And what kinds of messages resound with people the most?

#### Which emotions resonate most with Trump Twitter followers, and how do Trump's messages differ from his staffers' messages?

David Robinson wrote an [excellent analysis](http://varianceexplained.org/r/trump-tweets/) of Donald Trump tweets, concluding we can tell which messages come from Trump and which messages come from his campaign staff based on the device source. Trump uses Android, his staffers use iPhone, and based on Robinson's analysis there are clear differences in the behavior of the two device users. 

The question for me is: how is message popularity affected by Trump vs. his staffers?

## Contents

#### Emotional analysis of text

NLP tools for analyzing and categorizing the emotional content of text aren't too common. Most do not extend beyond classification of positive, negative, and neutral, whereas we are looking for a wider range of emotions. There are a few academic research papers which found moderate success with classification of emotions using relatively full-bodied models (combining lexicon, syntax, etc.), although the code was not provided. 

I settled on using Saif Mohammad's [NRC emotion lexicon[(http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm), which contains a dictionary of words with their associated emotions as judged by Mechanical Turkers (if I recall correctly). If interested, [his site](http://saifmohammad.com/WebPages/ResearchInterests.html#EmotionAnalysis
) has some good resources to help gain an understanding of this sub-field.

More recently I've discovered [IBM's Alchemy Language API](http://www.ibm.com/watson/developercloud/alchemy-language/api/v1/#introduction
) which, based on the demo, does a great job of analyzing emotional content in text. If I have time to refine my model, I'll take a look at incorporating this. 

#### Approach

- get_tweets.ipynb uses Twython to scrape tweets and stores them in MongoDb
- tweet_analysis.ipynb cleans the data and includes, among other things, feature engineering, Twitter-specific tokenization (emoticons, hashtags, URLs, etc.), token-matching against 3rd party dictionary, lots of analysis and graphs, stemming, topic modeling with td-idf and k-means, LDA modeling.

Conclusion to come...
