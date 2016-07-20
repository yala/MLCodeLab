# MLCodeLab

Welcome to MLCodeLab! Today we'll be building a sentiment analysis classifier for twitter. Our task is determining if a tweet is positive or negative.

In this lab we cover:

- Building a classifier end to end (from raw text to features to results!)
- Analysis tools and feature engineering
- Experimenting with different classifiers

In the talk, I'll walk through some sample code, but working through implementing the code yourself is heavily encouraged! It's not very tricky but generating the code yourself will solidify the knowledge. Let's get started!

# 1: Building it End to End  
In this section, we'll be building feature extraction, mapping tweets to a discrete sized array (a bag of words model), training a Perceptron and evaluating our results. The relevant python file is listed bellow.

- [model.py](model.py)


# 2: Analysis and Feature Engineering
In this section, we'll talk about a simple example of [underfitting and overfitting](http://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html). First we'll talk about learning curves, and how to get and intuition of what your learning algorithm is doing and how we can address each of those problems with our features. As before, the relevant file is listed below.

- [featureEngineering.py](featureEngineering.py)


# 3: Exploring more classifiers
Scikit-learn makes it really easy to experiment with different types of classifiers. In this section, we'll show how we can switch from the previous implemention of *model.py* with Perceptron to a Passive Agressive Classifier, or an SVM. The relevant python file is called:

- [allModels.py](allModels.py)

# Additional information:
One thing we didn't walkthrough is how we build the *.p* files. For completeness, I've included the code for that [here](data/buildDataSet.py).

### Sources
This twitter dataset for this code lab comes from [here](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/)

The full dataset with 1.5 million tweets can be downloaded [here](http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip)

More information about scikit-learn can be found at their [website](http://scikit-learn.org/)