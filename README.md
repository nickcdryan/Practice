# Projects
A selection of archived and WIP data science projects

### [Recurrent Neural Network Tutorial](https://github.com/nickcdryan/Projects/tree/master/neural_net_bot)
A how-to for anyone interested in training and sampling from their own state-of-the-art recurrent neural network. Intentionally designed so that anyone, with almost no familiarity of neural networks (or programming, for that matter), can get one up and running in an afternoon. Hopefully, this will allow people with very limited domain knowledge, but who are nevertheless interested in the latest advances in AI to gain access to the latest cutting-edge tool in the field.

Tutorial includes the output from two neural nets I trained and sampled from: 
- Cooking with HAL9000: AI generated recipes created from a net trained on 3MB of recipe text from cookbooks.com. Salmon Cookies, anyone?
- Donald Shakespeare: AI generated tweets and drama snippets created from a 2MB combination of the collected works of Shakespeare and Donald Trump tweets and speeches. Juliet said it best: "Here well get for me thy woo ask our see that speedy? #MakeAmericaGreatAgain"

### [Wine Quality](https://github.com/nickcdryan/Projects/tree/master/wine_quality)
What started as an analysis of wine datasets from the UCI repository has essentially become a library of data visualization, sklearn algorithm, and preprocessing snippets that I draw from in order to get a fast, first-pass understanding of new datasets.

Includes a wide range of tools, from violinplots, parallel coordinates, and PCA biplots to elastic net regression, adaboost, and KNN to confusion matrices, gridsearch, and scaling tools.

### [Trump Tweet Analysis](https://github.com/nickcdryan/Projects/tree/master/trump_analysis)
Semantic analysis of Trump tweets using Twitter API, MongoDB, NLP packages, and emotion lexicon. The purpose is to understand the relation between the type of emotion conveyed in a tweet vs. the popularity of that tweet in order to understand what Trump twitter followers respond to.

Includes script for pulling tweets via the API and Twython and storing in MongoDB, feature engineering, Twitter-specific tokenization (emoticons, hashtags, URLs, etc.), token-matching against 3rd party dictionary, lots of analysis and graphs, stemming, topic modeling with td-idf and k-means, LDA modeling.

### [Income Analysis](https://github.com/nickcdryan/Projects/tree/master/income_analysis_project) 
Prediction and analysis of individual yearly income using US census data achieving 98.5% accuracy with random forest classifier in scikit-learn. 

While not the most demanding classification task, this project includes careful analysis of a messy dataset, in-depth discussion of tools, and does a good job of demonstrating my thought process through a careful analysis of messy data and discussion of tools. Discussion of handling null values, data gaps, preprocessing, encoding methods for categorical data, interpretation of regression coefficients, interpreting scores, confusion matrices, diagnostics for model improvement.

### [Technical Writing](https://github.com/nickcdryan/Projects/tree/master/technical_writing)
Contains a couple samples of more technical or mathematically involved topics. The regularization write-up aims to summarize the purpose of regularization, the difference between types of regularization used, the mathematical consequences of using one type of regularization over another, and some loose guidelines about which to use. 

### [Monte Carlo Project Plan Simulator](https://github.com/nickcdryan/MonteCarlo_Estimate)
Takes in a number of tasks to be completed for a given project, the hours estimated for each task, and the confidence that the task will be completed with those hours. Uses Monte Carlo simulation to create a cumulative distribution function showing the likelihood for the entire project to be completed within a given time frame. 

Future work would record actual performance, score and store underestimation and overestimation associated with given types of task, individuals, and disciplines, and finally would use this to intelligently adjust estimations for future projects. E.g. John always underestimates the time taken to complete a task by roughly 20%, everyone always overestimates design time by 15%, etc.

