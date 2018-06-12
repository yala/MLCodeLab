import pickle
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from sklearn.learning_curve import learning_curve
import numpy as np

trainSetPath = "data/train.p"
devSetPath   = "data/dev.p"


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


def evaluate(trainX, trainY, devX, devY, n_iter, prefix=""):
    perceptron = Perceptron(n_iter=n_iter)
    perceptron.fit(trainX, trainY)
    # Predict labels for devSet
    perceptron.predict(devX)
    #Print out accuracy for trainSet
    print(prefix, "Train set accuracy:", perceptron.score(trainX, trainY))
    #Print out accuracy for devSet
    print(prefix, "Dev set accuracy:", perceptron.score(devX, devY))
    print("--")

if __name__ == "__main__":

    trainSet =  pickle.load(open(trainSetPath, 'rb'), encoding='bytes')

    #Extract tweets and labels into 2 lists
    trainTweets = [t[0] for t in trainSet]
    trainY = [t[1] for t in trainSet]

    countVec = CountVectorizer(max_features=5000, ngram_range=(1,1))
    countVec.fit(trainTweets)
    dictVec  = DictVectorizer()

    #Get features and labels for development set.
    devSet = pickle.load(open(devSetPath, 'rb'), encoding='bytes')
    devTweets = [d[0] for d in devSet]

    devY = [d[1] for d in devSet]


    printVocab = True

    #### Start -- above is the same code as we saw in model.py, but a little less verbose. Now let's dig deeper into
    #### The shoddy performance and use our insights to design new features, and iterate to a better model.

    # ITERATION 1

    # Let's see what happens when we let the model train for longer.
    print("Iteration 1 - Training iters")
    print("#####################")
    if printVocab:
        print("countVec vocab")
        print(countVec.get_feature_names())
    trainX = getFeatures(trainTweets, countVec, dictVec, True)
    devX = getFeatures(devTweets,  countVec, dictVec)

    evaluate(trainX, trainY, devX, devY, 15, "15 iters")
    evaluate(trainX, trainY, devX, devY, 150, "150 iters")
    evaluate(trainX, trainY, devX, devY, 250, "250 iters")




    # ITERATION 2
    # Let's make the model more expressive by increasing the vocab size
    print("Iteration 2 - Vocab size")
    print("#####################")
    maxVocabSize = 10000
    countVecLargeVocab = CountVectorizer(max_features=maxVocabSize)
    countVecLargeVocab.fit(trainTweets)
    if printVocab:
        print("Large vocab")
        print(countVecLargeVocab.get_feature_names())

    trainXLargeVocab = getFeatures(trainTweets, countVecLargeVocab, dictVec)
    devXLargeVocab= getFeatures(devTweets,  countVecLargeVocab, dictVec)
    evaluate(trainXLargeVocab, trainY, devXLargeVocab, devY, 250, "LargeVocab")


    # ITERATION 3
    # Let's try filtering out stop words
    print("Iteration 3 - Stop words")
    print("#####################")
    maxVocabSize = 10000
    countVecStopWords = CountVectorizer(max_features=maxVocabSize, stop_words="english")
    countVecStopWords.fit(trainTweets)
    if printVocab:
        print( "Large vocab after filitering out stop_words")
        print( countVecStopWords.get_feature_names())

    trainXStopWords = getFeatures(trainTweets, countVecStopWords, dictVec)
    devXStopWords= getFeatures(devTweets,  countVecStopWords, dictVec)
    evaluate(trainXStopWords, trainY, devXStopWords, devY, 250, "StopWords")


    # ITERATION 4
    # Let's try including ngrams
    print("Iteration 4 - Ngrams")
    print("#####################")
    ngramRange = (1,3)
    countVecNGram = CountVectorizer(max_features=maxVocabSize, ngram_range=ngramRange)
    countVecNGram.fit(trainTweets)
    if printVocab:
        print("Large vocab after filitering out stop_words")
        print(countVecNGram.get_feature_names())

    trainXNGram = getFeatures(trainTweets, countVecNGram, dictVec)
    devXNGram = getFeatures(devTweets,  countVecNGram, dictVec)

    evaluate(trainXNGram, trainY, devXNGram, devY, 250, "Ngrams")

