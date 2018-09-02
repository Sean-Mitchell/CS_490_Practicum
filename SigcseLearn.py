import re
import os
import numpy as np
import pandas as pd
import string as strng
from scipy.sparse import hstack, metrics, svm
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, stop_words
from sklearn.model_selection import KFold, StratifiedKFold


#Thank you Internet
#https://stackoverflow.com/questions/16869990/how-to-convert-from-boolean-array-to-int-array-in-python
def boolstr_to_floatstr(v):
    if v == 'True':
        return '1'
    elif v == 'False':
        return '0'
    else:
        return '0'


# Read in all the txt Files
def LoopThroughDocuments(filePath, folderName):
    fileNames= os.listdir(filePath)
    dataframe = pd.DataFrame(columns=['FileName','RawText'])
    
    # Don't worry about reading files in if there is no summary atm
    if 'summary.txt' not in fileNames:
        return dataframe        
        
    # used for index creation while adding into a new dataframe
    counter = 0
    
    # loop through all the files in the folder
    for fileName in fileNames: 
    
        f = open(os.path.join(os.path.abspath(filePath), fileName), 'r', encoding='ISO-8859-1')
        
        # Read file and split based by sentences
        # remove new lines, split on strings that have a "." plus any white space, or split on ?!; or .* plus -(2 or more dashes) or . word whitespace
        rawText = f.read().lower()
        
        # Remove Stop Words
        for stopword in stop_words.ENGLISH_STOP_WORDS:
            rawText = re.sub(r'\b' + stopword.lower() + r'\b', '', rawText)
        
        rawText = re.split(r'\.\s+|[?!;]|\.*\-{2,}|\.\w\s|,\n+\s*', rawText)
        rawText = [string for string in rawText if ' ' in string]
        rawText = [string.strip() for string in rawText]
        rawText = [re.sub('[\n]', r'', string) for string in rawText] 
        f.close()
        
        # Assigns the summary into the dataframe
        if fileName == 'summary.txt':            
            
            # Create dataframe and concat it to what exists (if something exists)
            # Add all sentences into dataframe
            textObject = {'FileName' : folderName + '__summary', 'RawText' : rawText, 'NoPunctuation' : ''}    
                
        # Checks to see if the text file is a number and if it is read it into the main dataframe
        elif fileName.split('.')[0].isnumeric():
            
            # Create dataframe and concat it to what exists (if something exists)
            # Add all sentences into dataframe
            textObject = {'FileName' : folderName + '__' + str(counter), 'RawText' : rawText, 'NoPunctuation' : ''}     
            counter += 1
        
        if dataframe.empty:
            dataframe = pd.DataFrame.from_dict(textObject)
        else:
            dataframe = pd.concat([dataframe, pd.DataFrame.from_dict(textObject)], ignore_index=True, sort=False)
            
    dataframe.reset_index(drop = False)
    for index, row in dataframe.iterrows():
        dataframe.iloc[index]['NoPunctuation'] = ''.join(ch for ch in row['RawText'] if ch not in strng.punctuation)
                

    return dataframe
    
    
# Split the dataframes into the final data frame that will be used
# This includes matching up the summary sentences, vectorizing, and tfidf as well as assigning 
def ModifyRawData(rawDataFrame, rawEmails, rawSummaries):
    
    #print(rawEmails.head())
    #print(rawSummaries.head())
    #print(rawEmails.iloc[0]['NoPunctuation'])
    #print(rawEmails.shape)    
    #countVectorText = vect.fit_transform(rawEmails['NoPunctuation'])
    #print(countVectorText.shape)
    
    # Create Y column (this is what we will be working to get using the SVM later on).  It's the unknown we want to solve for later on    
    rawEmails.reset_index(drop = False)
    summaryList = rawEmails['RawText'].isin(rawSummaries['RawText'])
    
    #print(summaryList)
    print(MachineLearningPart(rawEmails, summaryList))
    
    #return dataframe    

#This should hold all the machine learning things
def MachineLearningPart(Emails, IsGoodSentenceList):
    vect = CountVectorizer(ngram_range=(1, 2))
    tfidfVect = TfidfVectorizer(ngram_range=(1, 2))
    
    #Assign Test and Train parts
    emails = Emails['NoPunctuation']
    goodSentences = IsGoodSentenceList
    
    #Create Series that the countVectorizer can use to create training and testing sets
    emails_train, emails_test, goodSentences_train, goodSentences_test = train_test_split(emails, goodSentences, random_state=1)
    vect.fit(emails)
    tfidfVect.fit(emails)
    # fit and transform training into vector matrix
    vect_emails_train_dtm = vect.transform(emails_train)
    tfid_emails_train_dtm = tfidfVect.transform(emails_train)
    
    # transform test into test matrix
    vect_emails_test_dtm = vect.transform(emails_test)
    tfid_emails_test_dtm = tfidfVect.transform(emails_test)
    
    # Scale train and test vector sets
    maxVal = vect_emails_train_dtm.max()
    vect_emails_train_dtm = vect_emails_train_dtm/float(maxVal)
    maxVal = tfid_emails_train_dtm.max()
    tfid_emails_train_dtm = tfid_emails_train_dtm/float(maxVal)
    
    # Concatonate the columns of the training and test set
    vect_tfidf_emails_train_dtm = hstack([vect_emails_train_dtm, tfid_emails_train_dtm])
    vect_tfidf_emails_test_dtm = hstack([vect_emails_test_dtm, tfid_emails_test_dtm])
    
    #Double Check shapes
    #print(emails_train_dtm)
    #print(emails_train_dtm.shape)
    #print(emails_test.shape)
    #print(emails_test_dtm.shape)
    #print(tfid_emails_test_dtm.shape)
    #print(vect_tfidf_emails_train_dtm.shape)
    #print(vect_tfidf_emails_train_dtm)
    
    # This prints off indices of true values
    #print([i for i, x in enumerate(goodSentences_train) if x])
    
    #Set up svc stuff (will modify into loop later)
    clf = svm.SVC(C=1.0, cache_size=8000, class_weight=None, coef0=0.1,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=1, shrinking=True,
    tol=.01, verbose=False)

    clf.fit(tfid_emails_train_dtm, goodSentences_train)    
    
    vect_tfidf_emails_results = clf.predict(tfid_emails_test_dtm)
    #print the accuracy is
    print('CountVectorizer + TFIDFVectorizer Results: ')
    print(metrics.accuracy_score(goodSentences_test, vect_tfidf_emails_results))    
    print(metrics.precision_recall_fscore_support(goodSentences_test, vect_tfidf_emails_results))    
    #This is a thing.  I am still uncertain how to use it
    print(metrics.confusion_matrix(goodSentences_test, vect_tfidf_emails_results))
        
    accuracy_Array = []
    accuracy_Array.append(metrics.accuracy_score(goodSentences_test, vect_tfidf_emails_results))
    
    
    '''
    randomStateCount = 1
    for cAmount in np.linspace(0.1,1,10):
            for coef0Amount in np.linspace(0,1,11):
                for tolAmount in [.001, .01, .1]:                    
                    
                    #Set up svc stuff (will modify into loop later)
                    clf = svm.SVC(C=cAmount, cache_size=8000, class_weight=None, coef0=coef0Amount,
                    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                    max_iter=-1, probability=False, random_state=randomStateCount, shrinking=True,
                    tol=tolAmount, verbose=False)

                    clf.fit(vect_tfidf_emails_train_dtm, goodSentences_train)    
                    
                    vect_tfidf_emails_results = clf.predict(vect_tfidf_emails_test_dtm)
                    #print the accuracy is
                    print('CountVectorizer + TFIDFVectorizer Results: ')
                    print(metrics.accuracy_score(goodSentences_test, vect_tfidf_emails_results))    
                    #This is a thing.  I am still uncertain how to use it
                    print(metrics.confusion_matrix(goodSentences_test, vect_tfidf_emails_results))
                        
                    accuracy_Array = []
                    accuracy_Array.append(metrics.accuracy_score(goodSentences_test, vect_tfidf_emails_results))
                    
                    randomStateCount += 1
                    
    goodSentences = IsGoodSentenceList
    #initialize folds'''
    kf = KFold(n_splits = 10, shuffle = True, random_state = 45)
    '''
    #The internet told me to split it like this
    for train_index, test_index in kf.split(emails, goodSentences):
        print('here')
        print(goodSentences[train_index].max)
        
        # fit and transform training into vector matrix
        emails_train_dtm = vect.fit_transform(emails.iloc[train_index].values)
        emails_test_dtm = vect.transform(emails.iloc[test_index].values)
        
        print(goodSentences[train_index].max)
        
        #goodSentences_train = vect.transform(goodSentences[goodSentences].values)
        print(goodSentences[train_index])
        goodSentences_train = np.vectorize(boolstr_to_floatstr)(goodSentences[train_index]).astype(int)
        print(goodSentences_train)
        
        
        clf = svm.SVC()
        clf.fit(emails_train_dtm, goodSentences_train)    
        #Set up svc stuff (will modify into loop later)
        svm.SVC(C=1.0, cache_size=400, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
        #see how it got split up
        #print (len(train_index), len(test_index))
        
        
        #Fit and then compare the predictions
        emails_results = clf.predict(emails_test_dtm)
        
        #Assign to return array and print
        accuracy_Array.append(metrics.accuracy_score(metrics.accuracy_score(goodSentences[test_index].values, emails_results)))
        print(metrics.accuracy_score(goodSentences[test_index].value, emails_results))
        print(metrics.confusion_matrix(goodSentences[test_index].value, emails_results))'''
    
    
    '''
    #Bring in NaiveBayes and apply it to the test set
    #This is the actual machine learning part
    nb = MultinomialNB()
    nb.fit(emails_train_dtm, goodSentences_train)
    category_prediction_test = nb.predict(emails_test_dtm)
    
    #print out what the categories are
    print(goodSentences_test.unique())
    
    #print the accuracy is
    print(metrics.accuracy_score(goodSentences_test, category_prediction_test))    
    #This is a thing.  I am still uncertain how to use it
    print(metrics.confusion_matrix(goodSentences_test, category_prediction_test))

    #Create and assign the start of the return array the answer for the first accuracy score
    accuracy_Array = []
    accuracy_Array.append(metrics.accuracy_score(goodSentences_test, category_prediction_test))
    
    
    #initialize folds
    kf = KFold(n_splits = 10, shuffle = True, random_state = 45)
    print(kf.get_n_splits(emails_train, goodSentences_train))
    
    #The internet told me to split it like this
    for train_index, test_index in kf.split(emails, emails):
        #Create a new naive_bayes model for each test set and then put its accuracy in an array
        nb = MultinomialNB()
        
        #see how it got split up
        #print (len(train_index), len(test_index))
        
        # fit and transform training into vector matrix
        emails_train_dtm = vect.fit_transform(emails.iloc[train_index].values)
        emails_test_dtm = vect.transform(emails.iloc[test_index].values)
        
        #Fit and then compare the predictions
        nb.fit(emails_train_dtm, goodSentences.iloc[train_index].values)
        category_prediction_test = nb.predict(emails_test_dtm)
        
        #Assign to return array and print
        accuracy_Array.append(metrics.accuracy_score(goodSentences.iloc[test_index].values, category_prediction_test))
        print(metrics.accuracy_score(goodSentences.iloc[test_index].values, category_prediction_test))    
        print(metrics.confusion_matrix(goodSentences.iloc[test_index].values, category_prediction_test))
    '''
    return accuracy_Array

def main():    
    '''
    #Get before shape
    #print(concatdf.shape)
    
    #Drop NaN values 
    concatdf = concatdf.dropna()
    
    #Get Results (first in array is the accuracy from the split)
    #All else are accuracies from the kfold split
    nb_accuracy = MachineLearningPart(concatdf)
    
    
    print('Split Accuracy: ' + str(nb_accuracy[0]))
    print('Kfold Accuracy Ave: ' + str(sum(nb_accuracy[1:])/len(nb_accuracy[1:])))
    '''

    # ###########################################################
    #            Read in Files and create rawDataFrame            #
    # ###########################################################
    
    filepath = 'Sigcse/'
    df = pd.DataFrame(columns=['FileName','RawText'])
    filenames= os.listdir(filepath)
    result = []
    for filename in filenames: # loop through all the files and folders
    
        if os.path.isdir(os.path.join(os.path.abspath(filepath), filename)): # check whether the current object is a folder or not    
            # if empty insert into the new dataframe
            if df.empty:
                df = LoopThroughDocuments(os.path.join(os.path.abspath(filepath), filename), filename)
            else:
                df = pd.concat([df,LoopThroughDocuments(os.path.join(os.path.abspath(filepath), filename), filename)], ignore_index=True, sort=False)
    
    
    # #################################################################
    #            Modify Raw data into modified data vector            #
    #                     (will be normalized later)                  #
    # #################################################################    
    
    #revisedDateFrame = 
    ModifyRawData(df,  df[df['FileName'].str.contains('summary')==False], df[df['FileName'].str.contains('summary')])
    # prints full head
    # pd.set_option('display.max_colwidth', -1)
    # print(df.head())
    
    
main()

#Emails prefaced by “from email:”

#Sample GT code
'''import nltk,random,pickle
from nltk.corpus import movie_reviews, stopwords

def doc_features(document,word_features):
    doc_words=set(document)
    features={}
    for word in word_features:
            features['contains(%s)' % word]=(word in doc_words)
    return features


def main():
    docs = [(list(movie_reviews.words(fileid)),category) for category in movie_reviews.categories() 
            for fileid in movie_reviews.fileids(category)]

    random.shuffle(docs)

    stop = stopwords.words('english')
    words = [w.lower() for w in movie_reviews.words() if w.lower() not in stop]
    all_words=nltk.FreqDist(words)
    word_features=list(all_words.keys())[:2000]
    print(word_features)
    
    featuresets = [(doc_features(d,word_features),c) for (d,c) in docs]
    print(featuresets[1])
    
    train_set = featuresets[100:]
    test_set = featuresets[:100]

    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print (nltk.classify.accuracy(classifier, test_set))

    classifier.show_most_informative_features(20)
    saveClassifier('movieNB.pickle',classifier)
    
def saveClassifier(name, classifier):
    f = open(name, 'wb')
    pickle.dump(classifier, f)
    f.close()

def loadClassifier(name):
    f = open(name)
    classifier = pickle.load(f)
    f.close()
    return classifier
    
main()
'''