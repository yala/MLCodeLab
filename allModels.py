import pickle as p
from sklearn.linear_model import Perceptron
from sklearn.linear_model  import PassiveAggressiveClassifier
from sklearn.svm          import SVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import numpy as np

trainSetPath = "data/train.p"
devSetPath = "data/dev.p"
testSetPath = "data/test.p"

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

if __name__ == "__main__":

    trainSet =  p.load(open(trainSetPath, 'rb'))
    devSet =  p.load(open(devSetPath, 'rb'))
    testSet =  p.load(open(testSetPath, 'rb'))


    perceptron = Perceptron(n_iter=15)
    passAgg    = PassiveAggressiveClassifier(n_iter=350)

    #Note SVM's time complexity grows quadratically as the size of the training set increases
    #SVM cannot easily scale past 10,000 samples (source sklearn), so for the SVM, we'll be feading in a truncated train set of 10,000.
    svm        = SVC()


    #Extract tweets and labels into 2 lists
    trainTweets = [t[0] for t in trainSet]
    trainY = [t[1] for t in trainSet]

    trainTweetsSmall = trainTweets[:10000]
    trainYSmall = trainY[:10000]

    devTweets = [t[0] for t in devSet]
    devY = [t[1] for t in devSet]


    testTweets = [t[0] for t in testSet]
    testY = [t[1] for t in testSet]
    
    dictVec  = DictVectorizer()


    # ITERATION 4 from previous section

    ngramRange = (1,3)
    maxVocabSize = 1000
    countVecNGram = CountVectorizer(max_features=maxVocabSize, ngram_range=ngramRange )
    countVecNGram.fit(trainTweets)


    trainX = getFeatures(trainTweets, countVecNGram, dictVec, True)
    trainXSmall = getFeatures(trainTweetsSmall, countVecNGram, dictVec)


    devX = getFeatures(devTweets, countVecNGram, dictVec)
    testX = getFeatures(testTweets, countVecNGram, dictVec)

    percScoresTrain = []
    percScoresDev = []
    for i in range(10):
        perceptron.fit(trainX, trainY)
        percScoresDev.append(perceptron.score(devX, devY))
        percScoresTrain.append(perceptron.score(trainX, trainY))

    print "Perceptron Train:", np.mean(percScoresTrain)
    print "Perceptron Dev:", np.mean(percScoresDev)
    
    passAggScoresTrain = []
    passAggScoresDev = []
    for i in range(10):
        passAgg.fit(trainX, trainY) 
        passAggScoresDev.append( passAgg.score(devX, devY))
        passAggScoresTrain.append( passAgg.score(trainX, trainY))


    print "Passive Aggressive Train:", np.mean(passAggScoresTrain)
    print "Passive Aggressive Dev:", np.mean(passAggScoresDev)

    
    passAggScoresSmallTrain = []
    passAggScoresSmallDev = []
    for i in range(10):
        passAgg.fit(trainX, trainY) 
        passAggScoresSmallDev.append( passAgg.score(devX, devY))
        passAggScoresSmallTrain.append( passAgg.score(trainXSmall,trainYSmall))

    print "Passive Aggressive (Small Dataset)) Train:", np.mean(passAggScoresSmallTrain)
    print "Passive Aggressive (Small Dataset)) Dev:", np.mean(passAggScoresSmallDev)

    svm.fit(trainXSmall, trainYSmall)
    print "SVM Train (Small Dataset):", svm.score(trainXSmall, trainYSmall)
    print "SVM Dev:", svm.score(devX, devY)
   

    #Now we use the best scoring model on dev, to select our final model, which run on the heldout set, test.
    # In our case, it's our Perceptron model

    print "Perceptron on Test, final result", perceptron.score(testX, testY)


   