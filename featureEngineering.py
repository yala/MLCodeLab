import pickle as p
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
import numpy as np

trainSetPath = "data/train.p"

# For a given list tweet (in raw text), return a matrix of features 
#(one row representing each tweet).
# If we are getting features for the trainSet, have the dictVectorizer learn
# how to index additional features 
def getFeatures(tweets, countVec, dictVec, isTrainSet=False):
    bagOfWordsMatrix = countVec.transform(tweets)
    # Build list of feature dictionaries
    additionalFeatures = []
    for tweet in tweets:
        additFeature = {'tweetLength':len(tweet)}
        additionalFeatures.append(additFeature)
    if isTrainSet:
        additionalFeatureMatrix = dictVec.fit_transform(additionalFeatures)
    else:
        additionalFeatureMatrix = dictVec.transform(additionalFeatures)

    # Horizontally stack matrices (column-wise) to build one large feature matrix
    X = hstack([bagOfWordsMatrix,additionalFeatureMatrix])    

    return X


def plot(train_sizes, train_scores, test_scores, title):
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    #Setup plot
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title(title)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.plot(train_sizes, [.5 for t in test_scores_mean], 'o-', color="b",
             label="Random")
    plt.legend(loc="best")

    #Display plot
    plt.show()

if __name__ == "__main__":

    trainSet =  p.load(open(trainSetPath, 'rb'))

    perceptron = Perceptron(n_iter=15)

    #Extract tweets and labels into 2 lists
    trainTweets = [t[0] for t in trainSet]
    trainY = [t[1] for t in trainSet]
    
    countVec = CountVectorizer(max_features=100)
    countVec.fit(trainTweets)
    dictVec  = DictVectorizer()


    printVocab = False

    #### Start -- above is the same code as we saw in model.py, but a little less verbose. Now let's dig deeper into 
    #### The shoddy performance. We'll use our insights to design new features, and iterate to a better model.

    # ITERATION 1

    #We'll be doing this with the use of learning curves
    #The learning curve method varies the size of how much data we use to train, and shows us our train and validation 
    # error.
    #Since it does this across a cv runs, for each size, we have cv scores, and we find their mean and stds for plotting
    #The sizes here are prop to the full size of the train set 
    sizes = [.01, .1, .2, .33, .55, .66, .8]  
    print "Iteration 1"
    if printVocab:
        print "countVec vocab"
        print countVec.get_feature_names()
    trainX = getFeatures(trainTweets, countVec, dictVec, True)
    train_sizes1, train_scores1, test_scores1 = learning_curve(perceptron, trainX, trainY, train_sizes=sizes, cv=10)

    print "Orig- Dev set accuracy:", np.mean(test_scores1[-1])
    plot(train_sizes1, train_scores1, test_scores1, "Original Model")
    print

    # ITERATION 2
    # Let's adding some regularization (since our feature norms are very large) and
    # increase our vocab size, since it doesn't seem very meaningful
    print "Iteration 2"
    maxVocabSize = 1000
    countVecLargeVocab = CountVectorizer(max_features=maxVocabSize)
    countVecLargeVocab.fit(trainTweets)
    if printVocab:
        print "Large vocab"
        print countVecLargeVocab.get_feature_names()

    trainXLargeVocab = getFeatures(trainTweets, countVecLargeVocab, dictVec)
    train_sizes2, train_scores2, test_scores2 = learning_curve(perceptron, trainXLargeVocab, trainY, train_sizes=sizes, cv=10)

    print "Large Vocab- Dev set accuracy:", np.mean(test_scores2[-1])
    plot(train_sizes2, train_scores2, test_scores2, "Large Vocab")
    print

    # ITERATION 3
    # Let's try filtering out stop words
    print "Iteration 3"
    countVecStopWords = CountVectorizer(max_features=maxVocabSize, stop_words="english")
    countVecStopWords.fit(trainTweets)
    if printVocab:
        print "Large vocab after filitering out stop_words"
        print countVecStopWords.get_feature_names()

    trainXStopWords = getFeatures(trainTweets, countVecStopWords, dictVec)
    train_sizes3, train_scores3, test_scores3 = learning_curve(perceptron, trainXStopWords, trainY, train_sizes=sizes, cv=10)

    print "Large Vocab+Stopwords- Dev set accuracy:",  np.mean(test_scores3[-1])
    plot(train_sizes3, train_scores3, test_scores3, "Large Vocab and Stop words")
    print


    # ITERATION 4
    # Let's try including ngrams
    print "Iteration 4"
    ngramRange = (1,3)
    countVecNGram = CountVectorizer(max_features=maxVocabSize, ngram_range=ngramRange )
    countVecNGram.fit(trainTweets)
    if printVocab:
        print "Large vocab after filitering out stop_words"
        print countVecNGram.get_feature_names()

    trainXNGram = getFeatures(trainTweets, countVecNGram, dictVec)
    train_sizes4, train_scores4, test_scores4 = learning_curve(perceptron, trainXNGram, trainY, train_sizes=sizes, cv=10)

    print "Large Vocab+nGrams",ngramRange,"- Dev set accuracy:",  np.mean(test_scores4[-1])
    plot(train_sizes4, train_scores4, test_scores4, "Large Vocab and nGrams "+str(ngramRange))
    print


    # ITERATION 5 - Overfit
    # Let's try including increasing our ngrams
    print "Iteration 5"
    ngramRange = (1, 10)
    countVecNGram = CountVectorizer(max_features=maxVocabSize, ngram_range=ngramRange )
    countVecNGram.fit(trainTweets)
    if printVocab:
        print "Large vocab after filitering out stop_words"
        print countVecNGram.get_feature_names()

    trainXNGram = getFeatures(trainTweets, countVecNGram, dictVec)

    train_sizes5, train_scores5, test_scores5 = learning_curve(perceptron, trainXNGram, trainY, train_sizes=sizes, cv=10)

    print "Large Vocab+nGrams",ngramRange,"- Dev set accuracy:",  np.mean(test_scores5[-1])
    plot(train_sizes5, train_scores5, test_scores5, "Large Vocab and nGrams "+str(ngramRange))
    print
    
    # ITERATION 6 - Regularization, fight the overfit
    # Let's try adding some regularization (since our feature norms are very large) 
    print "Iteration 6"
    perceptronReg = Perceptron(penalty="l1", n_iter=15)
    train_sizes6, train_scores6, test_scores6 = learning_curve(perceptronReg, trainXNGram, trainY, train_sizes=sizes, cv=10)

    print "Large Vocab+nGrams",ngramRange," with L1 Reg- Dev set accuracy:", np.mean(test_scores6[-1])
    
    plot(train_sizes6, train_scores6, test_scores6, "Regularization")
    print

