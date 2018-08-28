import re
import os
import string as strng
import pandas as pd
import numpy as np
#from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold

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
        # Assigns the summary into the dataframe
        if fileName == 'summary.txt': 
            
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
            #rawText = [re.sub(str.punctuation, r'', string) for string in rawText]
            f.close()
            
            # Create dataframe and concat it to what exists (if something exists)
            # Add all sentences into dataframe
            textObject = {'FileName' : folderName + '__summary', 'RawText' : rawText, 'NoPunctuation' : ''}    
            if dataframe.empty:
                dataframe = pd.DataFrame.from_dict(textObject)
            else:
                dataframe = pd.concat([dataframe, pd.DataFrame.from_dict(textObject)], ignore_index=True, sort=False)
                
        # Checks to see if the text file is a number and if it is read it into the main dataframe
        elif fileName.split('.')[0].isnumeric():
        
            f = open(os.path.join(os.path.abspath(filePath), fileName), 'r', encoding='ISO-8859-1')   
            # Read file and split based by sentences
            # remove new lines, split on strings that have a "." plus any white space, or split on ?!; or multiline ---- elements
            rawText = f.read().lower()            
            
            # Remove Stop Words
            for stopword in stop_words.ENGLISH_STOP_WORDS:
                rawText = re.sub(r'\b' + stopword.lower() + r'\b', '', rawText)
            
            rawText = re.split(r'\.\s+|[?!;]|\.*\-{2,}|\.\w\s|,\n+\s*', rawText)
            rawText = [string for string in rawText if ' ' in string]
            rawText = [string.strip() for string in rawText]
            rawText = [re.sub('[\n]', r'', string) for string in rawText] 
            f.close()
            
            # Create dataframe and concat it to what exists (if something exists)
            # Add all sentences into dataframe
            textObject = {'FileName' : folderName + '__' + str(counter), 'RawText' : rawText, 'NoPunctuation' : ''}       
            if dataframe.empty:
                dataframe = pd.DataFrame.from_dict(textObject)
            else:
                dataframe = pd.concat([dataframe, pd.DataFrame.from_dict(textObject)], ignore_index=True, sort=False)
            counter += 1
        
        #dataframe.reset_index(drop = False)
        '''for index, row in dataframe.iterrows():
            dataframe.iloc[index]['NoPunctuation'] = ''.join(ch for ch in row['RawText'] if ch not in strng.punctuation)
        '''        
    return dataframe
    
    
# Split the dataframes into the final data frame that will be used
# This includes matching up the summary sentences, vectorizing, and tfidf as well as assigning 
def ModifyRawData(rawDataFrame, rawEmails, rawSummaries):
    print('nice')
    vect = CountVectorizer()
    
    print(rawEmails.head())
    #print(rawSummaries.head())
    
    #nopunc = [char for char in rawEmails if char not in str.punctuation]
    '''rawEmails.reset_index(drop = False)
    print(rawEmails.iloc[0]['NoPunctuation'])
    print(rawEmails.head())
    
    countVectorText = vect.fit_transform(rawEmails)
    print(countVectorText.shape)
    
    # Create Y column (this is what we will be working to get using the SVM later on).  It's the unknown we want to solve for later on
    
    rawEmails['IsSummarySentence'] = rawEmails['RawText'].isin(rawSummaries['RawText'])
    '''
    #print(df.head())
    
    #return dataframe    

#This should hold all the machine learning things
def MachineLearningPart(dataFrame):
    vect = CountVectorizer()
    
    #Assign Test and Train parts
    descriptionX = dataFrame['Brief Description']
    categoryY = dataFrame['Category']
    
    #Create Series that the countVectorizer can use to create training and testing sets
    Description_train, Description_test, category_train, category_test = train_test_split(descriptionX, categoryY, random_state=1)
    
    #Double Check shapes
    # print(Description_train.shape)
    # print(Description_test.shape)
    # print(category_train.shape)
    # print(category_test.shape)
    
    
    # fit and transform training into vector matrix
    Description_train_dtm = vect.fit_transform(Description_train)

    # transform test into test matrix
    Description_test_dtm = vect.transform(Description_test)
    
    #Bring in NaiveBayes and apply it to the test set
    #This is the actual machine learning part
    nb = MultinomialNB()
    nb.fit(Description_train_dtm, category_train)
    category_prediction_test = nb.predict(Description_test_dtm)
    
    #print out what the categories are
    print(category_test.unique())
    
    #print the accuracy is
    print(metrics.accuracy_score(category_test, category_prediction_test))    
    #This is a thing.  I am still uncertain how to use it
    print(metrics.confusion_matrix(category_test, category_prediction_test))

    #Create and assign the start of the return array the answer for the first accuracy score
    accuracy_Array = []
    accuracy_Array.append(metrics.accuracy_score(category_test, category_prediction_test))
    
    
    #initialize folds
    kf = KFold(n_splits = 10, shuffle = True, random_state = 45)
    print(kf.get_n_splits(Description_train, category_train))
    
    #The internet told me to split it like this
    for train_index, test_index in kf.split(descriptionX, descriptionX):
        #Create a new naive_bayes model for each test set and then put its accuracy in an array
        nb = MultinomialNB()
        
        #see how it got split up
        #print (len(train_index), len(test_index))
        
        # fit and transform training into vector matrix
        Description_train_dtm = vect.fit_transform(descriptionX.iloc[train_index].values)
        Description_test_dtm = vect.transform(descriptionX.iloc[test_index].values)
        
        #Fit and then compare the predictions
        nb.fit(Description_train_dtm, categoryY.iloc[train_index].values)
        category_prediction_test = nb.predict(Description_test_dtm)
        
        #Assign to return array and print
        accuracy_Array.append(metrics.accuracy_score(categoryY.iloc[test_index].values, category_prediction_test))
        print(metrics.accuracy_score(categoryY.iloc[test_index].values, category_prediction_test))    
        print(metrics.confusion_matrix(categoryY.iloc[test_index].values, category_prediction_test))
    
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