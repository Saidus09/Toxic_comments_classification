# Toxic_comments_classification
### This project analyzes real comments from Wikipedia discussion pages
Due to the large volume of files, they have to be downloaded directly from the competition on Kaggle(https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview)

### Setting the task
There were 312,735 real comments from Wikipedia discussion pages and 6 classes of toxicity comments:
* toxic - ordinary toxic
* severe toxic - highly toxic
* obscene - obscene
* threat - containing a threat
* insult - offensive
* identity hate - containing hatred of a person

It is necessary to solve the classification problem: for each comment, learn how to predict with what probabilities to which classes of toxicity it belongs. A comment may not relate to any class of toxicity, or it may relate to one or more classes of toxicity.

### Data processing
The data was downloaded from [the Kaggle competition page](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge ). The training sample consists of 159571 objects, and the test sample consists of 153164 objects. The classes are not balanced - less than 11% of the comments from the training sample relate to at least one of the toxicity classes. For each comment from the test sample, 6 target variables are predicted: the probability of attributing the comment to the appropriate toxicity class.

### Comment text processing
The length of the comment turned out to be an uninformative sign that reduces the quality of the model, so it is not used in the final model.

The text of each comment is divided into separate words by a regular expression, after which [lemmatization] occurs (http://www.nltk.org/api/nltk.stem.html#nltk.stem.wordnet .WordNetLemmatizer) using the appropriate tools from the [NLTK] library(https://www.nltk.org /). As a result, each comment is converted into a list of words. After that, stop words are deleted: all words from the same letter, as well as stop words from the [NLTK library list](http://www.nltk.org/book/ch02.html#code-unusual ) and [Wordcloud Library list](https://github.com/amueller/word_cloud/blob/master/wordcloud/stopwords ).

### Comment text analysis
For toxic, clean, and all comments from the training sample using the [NLTK] library tools (http://www.nltk.org/index.html ):
* The number of unique words has been calculated
* The most popular words were found
* A word frequency distribution corresponding to [Zipf's Law] is constructed (https://ru.wikipedia.org/wiki/%D0%97%D0%B0%D0%BA%D0%BE%D0%BD_%D0%A6%D0%B8%D0%BF%D1%84%D0%B0 )
* The most popular bigrams and trigrams are found

### Building a model
Each comment is converted to a list of words. To create a matrix of features of objects, use [TF-IDF](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html ) from the [Scikit-learn] library(http://scikit-learn.org/stable/index.html ). Selected parameters: min_df=4, max_df=1.0.

To predict each of the 6 target variables, [logistic регрессия](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html ) from the [Scikit-learn] library(http://scikit-learn.org/stable/index.html ). In all cases, the optimal parameters were: C=1, penalty='l2'. The quality metric is the average value of the area under the ROC curve for the 6 predicted target variables. The final metric value is 0.97503.
