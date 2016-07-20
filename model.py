import pickle as p
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

# We have 60000 train tweets, 20000 dev and 20000 test
# Each of these files is a list of (tweet, sentiment pairs)
trainSetPath = "data/train.p"
devSetPath   = "data/dev.p"
testSetPath  = "data/test.p"

# For a given list tweet (in raw text), return a matrix of features 
#(one row representing each tweet).
# If we are getting features for the trainSet, have the dictVectorizer learn
# how to index additional features 
def getFeatures(tweets, countVec, dictVec, isTrainSet=False, verbose = False):
    bagOfWordsMatrix = countVec.transform(tweets)
    if verbose:
        print "Shape of bag of word matrix", bagOfWordsMatrix.shape
        raw_input()

    # Build list of feature dictionaries
    additionalFeatures = []
    for tweet in tweets:
        additFeature = {'tweetLength':len(tweet)}
        additionalFeatures.append(additFeature)

    if isTrainSet:
        additionalFeatureMatrix = dictVec.fit_transform(additionalFeatures)
    else:
        additionalFeatureMatrix = dictVec.transform(additionalFeatures)
    if verbose:
        print "Shape of additionalFeatureMatrix", additionalFeatureMatrix.shape
        raw_input()
        # Print feature names
        print "Names of additional features", dictVec.get_feature_names()        
        raw_input()

    # Horizontally stack matrices (column-wise) to build one large feature matrix
    X = hstack([bagOfWordsMatrix,additionalFeatureMatrix])    

    if verbose:
        print "Shape of feature matrix", X.shape

        print "Example of tweet:", tweets[0]
        analyser = countVec.build_analyzer()        
        print "How this tweet was tokenized", analyser(tweets[0])
        print "Corresponding feature vector", X.toarray()[0]
    return X

# Return portion of Labels with postion of labels
def getLabelDist(Y):
    portionPos =  sum([int(y) for y in Y])*1./len(Y)
    return "Percent Labels Positive", str(portionPos)

if __name__ == "__main__":

    trainSet =  p.load(open(trainSetPath, 'rb'))

    perceptron = Perceptron(verbose=15, n_iter=15)

    #Extract tweets and labels into 2 lists
    trainTweets = [t[0] for t in trainSet]
    trainY = [t[1] for t in trainSet]

    print "Train label distribution", getLabelDist(trainY)

    # To reperest a tweet, we'll start with the following features
    # Bag of words for the 50 most common words (we'll use a CountVectorizer for this)
    # Length of the tweet in characters        
    countVec = CountVectorizer(max_features=1500)
    dictVec  = DictVectorizer()

    # Step 1: Fit the CountVectorizer to the trainTweets
    countVec.fit(trainTweets)


    # Step 2: Implement getFeautres() to return a feature matrix for any
    # list of tweets.

    #Now get train features.
    trainX = getFeatures(trainTweets, countVec, dictVec, True, False)
    
    perceptron.fit(trainX, trainY)

    #Get features and labels for development set.
    devSet = p.load(open(devSetPath, 'rb'))
    devTweets = [d[0] for d in devSet]

    devX = getFeatures(devTweets, countVec, dictVec)
    devY = [d[1] for d in devSet]

    print "Train label distribution", getLabelDist(devY)

    # Predict labels for devSet
    perceptron.predict(devX)

    #Print out scores for devSet
    print perceptron.score(devX, devY)