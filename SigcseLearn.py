import os
import numpy as np
import pandas as pd
import random
import re
import string as strng
import sys
import time
from scipy.sparse import hstack
from sklearn import metrics, svm
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from threading import Lock, Thread

# Global variables for holding the percentages
statsLock = Lock()
statsArray = []
naiveStatsArray = []
queryTFIDF = []
includeTFIDF = True
isRunPerThread = False



# Read in all the txt Files
def LoopThroughDocuments(filePath, folderName):
    fileNames= os.listdir(filePath)
    dataframe = pd.DataFrame(columns=['RawFileName', 'FileName','CleanText', 'CleanTextNoPunc', 'FirstSentence', 'SecondSentence', 'ThirdSentence', 'FourthSentence', 'FifthSentence',
        'TopOneSentence', 'TopTwoSentence', 'TopThreeSentence', 'TopFourSentence', 'TopFiveSentence', 'SentenceLengthBeforeStop', 'CosineSimilarity'])
    dataframeNoStop = pd.DataFrame(columns=['RawFileName', 'FileName','CleanText', 'CleanTextNoPunc', 'FirstSentence', 'SecondSentence', 'ThirdSentence', 'FourthSentence', 'FifthSentence',
        'TopOneSentence', 'TopTwoSentence', 'TopThreeSentence', 'TopFourSentence', 'TopFiveSentence', 'SentenceLengthBeforeStop', 'CosineSimilarity'])
    
    # Don't worry about reading files in if there is no summary atm
    if 'summary.txt' not in fileNames:
        return dataframe, dataframeNoStop    
    

    queryTFIDF.append(folderName)
    
    
    # used for index creation while adding into a new dataframe
    counter = 0
    
    # loop through all the files in the folder
    for fileName in fileNames: 
    
        f = open(os.path.join(os.path.abspath(filePath), fileName), 'r', encoding='ISO-8859-1')
        
        # Read file and make copy of the file
        rawText = f.read().lower()
        f.close()
        RawTextNoStopWords = rawText + ' ' # this makes it a deep copy
        
        
        # Remove Stop Words
        for stopword in stop_words.ENGLISH_STOP_WORDS:
            RawTextNoStopWords = re.sub(r'\b' + stopword.lower() + r'\b', '', RawTextNoStopWords)
        
        #SPlit strings, remove sentences without a space in them, strip ends, remove newLine characters
        # remove new lines, split on strings that have a "." plus any white space, or split on ?!; or .* plus -(2 or more dashes) or . word whitespace
        rawText = re.split(r'\.\s+|[?!;]|\.*\-{2,}|\.\w\s|,\n+\s*', rawText)
        RawTextNoStopWords = re.split(r'\.\s+|[?!;]|\.*\-{2,}|\.\w\s|,\n+\s*', RawTextNoStopWords)
        rawText = [string.strip() for string in rawText if ' ' in string]
        RawTextNoStopWords = [string.strip() for string in RawTextNoStopWords if ' ' in string]
        rawText = [re.sub('[\n]', r'', string) for string in rawText] 
        RawTextNoStopWords = [re.sub('[\n]', r'', string) for string in RawTextNoStopWords]
        
        # #########################################################################################
        #                               Get Word Count Of Sentence                                  #
        # #########################################################################################
        RawSentenceLength = []
        CleanSentenceLength = []
        
        for sentenceCount in range(0, len(rawText)):
            RawSentenceLength.append(len(rawText[sentenceCount].split()))

        for sentenceCount in range(0, len(RawTextNoStopWords)):
            CleanSentenceLength.append(len(RawTextNoStopWords[sentenceCount].split()))
        
        # If there are no valid sentences in email, continue loop
        if not RawSentenceLength:
            continue
        
        maxVal = max(RawSentenceLength)
        normalized_RawSentenceLength = [x / float(maxVal) for x in RawSentenceLength]
        maxVal = max(CleanSentenceLength)
        normalized_CleanSentenceLength = [x / float(maxVal) for x in CleanSentenceLength]
        
        # #########################################################################################
        #                       Get sentence relative position in the document                    #
        # #########################################################################################
        isFirstRaw = np.zeros(len(rawText))
        isSecondRaw = np.zeros(len(rawText))
        isThirdRaw = np.zeros(len(rawText))
        isFourthRaw = np.zeros(len(rawText))
        isFifthRaw = np.zeros(len(rawText))
        isFirstNoStop = np.zeros(len(RawTextNoStopWords))
        isSecondNoStop = np.zeros(len(RawTextNoStopWords))
        isThirdNoStop = np.zeros(len(RawTextNoStopWords))
        isFourthNoStop = np.zeros(len(RawTextNoStopWords))
        isFifthNoStop = np.zeros(len(RawTextNoStopWords))
        
        RawTopTwoSentence = np.zeros(len(rawText))
        RawTopThreeSentence = np.zeros(len(rawText))
        RawTopFourSentence = np.zeros(len(rawText))
        RawTopFiveSentence = np.zeros(len(rawText))
        CleanTopTwoSentence = np.zeros(len(RawTextNoStopWords))
        CleanTopThreeSentence = np.zeros(len(RawTextNoStopWords))
        CleanTopFourSentence = np.zeros(len(RawTextNoStopWords))
        CleanTopFiveSentence = np.zeros(len(RawTextNoStopWords))
        
        # Set up sentence locality count
        if len(rawText) < 5:
            for count in range(0, len(rawText)):
                if count == 0:
                    isFirstRaw[count] = 1   
                    RawTopTwoSentence[count] = 1 
                    RawTopThreeSentence[count] = 1 
                    RawTopFourSentence[count] = 1 
                    RawTopFiveSentence[count] = 1 
                elif count == 1:
                    isSecondRaw[count] = 1  
                    RawTopTwoSentence[count] = 1 
                    RawTopThreeSentence[count] = 1 
                    RawTopFourSentence[count] = 1 
                    RawTopFiveSentence[count] = 1         
                elif count == 2:
                    isThirdRaw[count] = 1  
                    RawTopThreeSentence[count] = 1 
                    RawTopFourSentence[count] = 1 
                    RawTopFiveSentence[count] = 1         
                else:
                    isFourthRaw[count] = 1
                    RawTopFourSentence[count] = 1 
                    RawTopFiveSentence[count] = 1   
        
        else:
            for count in range(0, 5):
                if count == 0:
                    isFirstRaw[count] = 1 
                    RawTopTwoSentence[count] = 1 
                    RawTopThreeSentence[count] = 1 
                    RawTopFourSentence[count] = 1 
                    RawTopFiveSentence[count] = 1          
                elif count == 1:
                    isSecondRaw[count] = 1  
                    RawTopTwoSentence[count] = 1 
                    RawTopThreeSentence[count] = 1 
                    RawTopFourSentence[count] = 1 
                    RawTopFiveSentence[count] = 1        
                elif count == 2:
                    isThirdRaw[count] = 1  
                    RawTopThreeSentence[count] = 1 
                    RawTopFourSentence[count] = 1 
                    RawTopFiveSentence[count] = 1        
                elif count == 3:
                    isFourthRaw[count] = 1    
                    RawTopFourSentence[count] = 1 
                    RawTopFiveSentence[count] = 1        
                else:
                    isFifthRaw[count] = 1
                    RawTopFiveSentence[count] = 1  
                    
        if len(RawTextNoStopWords) < 5:
            for count in range(0, len(RawTextNoStopWords)):
                if count == 0:
                    isFirstNoStop[count] = 1 
                    CleanTopTwoSentence[count] = 1    
                    CleanTopThreeSentence[count] = 1
                    CleanTopFourSentence[count] = 1
                    CleanTopFiveSentence[count] = 1
                elif count == 1:
                    isSecondNoStop[count] = 1 
                    CleanTopTwoSentence[count] = 1    
                    CleanTopThreeSentence[count] = 1
                    CleanTopFourSentence[count] = 1
                    CleanTopFiveSentence[count] = 1           
                elif count == 2:
                    isThirdNoStop[count] = 1 
                    CleanTopThreeSentence[count] = 1
                    CleanTopFourSentence[count] = 1
                    CleanTopFiveSentence[count] = 1              
                else:
                    isFourthNoStop[count] = 1 
                    CleanTopFourSentence[count] = 1
                    CleanTopFiveSentence[count] = 1        
        else:
            for count in range(0, 5):
                if count == 0:
                    isFirstNoStop[count] = 1 
                    CleanTopTwoSentence[count] = 1    
                    CleanTopThreeSentence[count] = 1
                    CleanTopFourSentence[count] = 1
                    CleanTopFiveSentence[count] = 1           
                elif count == 1:
                    isSecondNoStop[count] = 1
                    CleanTopTwoSentence[count] = 1    
                    CleanTopThreeSentence[count] = 1
                    CleanTopFourSentence[count] = 1
                    CleanTopFiveSentence[count] = 1            
                elif count == 2:
                    isThirdNoStop[count] = 1  
                    CleanTopThreeSentence[count] = 1
                    CleanTopFourSentence[count] = 1
                    CleanTopFiveSentence[count] = 1           
                elif count == 3:
                    isFourthNoStop[count] = 1 
                    CleanTopFourSentence[count] = 1
                    CleanTopFiveSentence[count] = 1           
                else:
                    isFifthNoStop[count] = 1
                    CleanTopFiveSentence[count] = 1    
        
        
        # Assign bit that states whether the sentence is the first, second, ... , fifth (We'll see if this makes a difference)
        
        # Assigns the summary into the dataframe
        if fileName == 'summary.txt':            
            
            # Create dataframe and concat it to what exists (if something exists)
            # Add all sentences into dataframe
            textObject = {'RawFileName': folderName, 'FileName' : folderName + '__summary', 'CleanText' : rawText , 'CleanTextNoPunc' : '', 'FirstSentence': isFirstRaw, 'SecondSentence': isSecondRaw, 
                'ThirdSentence': isThirdRaw, 'FourthSentence': isFourthRaw, 'FifthSentence': isFifthRaw, 'TopOneSentence': isFirstRaw, 'TopTwoSentence': RawTopTwoSentence, 'TopThreeSentence': RawTopThreeSentence,
                'TopFourSentence': RawTopFourSentence, 'TopFiveSentence': RawTopFiveSentence, 'SentenceLengthBeforeStop': normalized_RawSentenceLength, 'CosineSimilarity': 0}    
            textObjectNoStopWords = {'RawFileName': folderName, 'FileName' : folderName + '__summary', 'CleanText' : RawTextNoStopWords, 'CleanTextNoPunc' : '', 'FirstSentence': isFirstNoStop, 'SecondSentence': isSecondNoStop, 
                'ThirdSentence': isThirdNoStop, 'FourthSentence': isFourthNoStop, 'FifthSentence': isFifthNoStop, 'TopOneSentence': isFirstNoStop, 'TopTwoSentence': CleanTopTwoSentence, 'TopThreeSentence': CleanTopThreeSentence,
                'TopFourSentence': CleanTopFourSentence, 'TopFiveSentence': CleanTopFiveSentence, 'SentenceLengthBeforeStop': normalized_CleanSentenceLength, 'CosineSimilarity': 0}  
                
        # Checks to see if the text file is a number and if it is read it into the main dataframe
        elif fileName.split('.')[0].isnumeric():
            
            # Create dataframe and concat it to what exists (if something exists)
            # Add all sentences into dataframe
            # if rawtext is 0 for some reason replace with empty strings
            textObject = {'RawFileName': folderName, 'FileName' : folderName + '__' + str(counter), 'CleanText' : rawText , 'CleanTextNoPunc' : '', 'FirstSentence': isFirstRaw, 'SecondSentence': isSecondRaw, 
                'ThirdSentence': isThirdRaw, 'FourthSentence': isFourthRaw, 'FifthSentence': isFifthRaw, 'TopOneSentence': isFirstRaw, 'TopTwoSentence': RawTopTwoSentence, 'TopThreeSentence': RawTopThreeSentence,
                'TopFourSentence': RawTopFourSentence, 'TopFiveSentence': RawTopFiveSentence, 'SentenceLengthBeforeStop': normalized_RawSentenceLength, 'CosineSimilarity': 0}   
            textObjectNoStopWords = {'RawFileName': folderName, 'FileName' : folderName + '__' + str(counter),'CleanText' : RawTextNoStopWords, 'CleanTextNoPunc' : '', 'FirstSentence': isFirstNoStop, 'SecondSentence': isSecondNoStop, 
                'ThirdSentence': isThirdNoStop, 'FourthSentence': isFourthNoStop, 'FifthSentence': isFifthNoStop, 'TopOneSentence': isFirstNoStop, 'TopTwoSentence': CleanTopTwoSentence, 'TopThreeSentence': CleanTopThreeSentence,
                'TopFourSentence': CleanTopFourSentence, 'TopFiveSentence': CleanTopFiveSentence, 'SentenceLengthBeforeStop': normalized_CleanSentenceLength, 'CosineSimilarity': 0}
            counter += 1

        if dataframeNoStop.empty:
            dataframeNoStop = pd.DataFrame.from_dict(textObjectNoStopWords)    
        else:
            dataframeNoStop = pd.concat([dataframeNoStop, pd.DataFrame.from_dict(textObjectNoStopWords)], ignore_index=True, sort=False)
            
        if dataframe.empty:
            dataframe = pd.DataFrame.from_dict(textObject)
        else:
            dataframe = pd.concat([dataframe, pd.DataFrame.from_dict(textObject)], ignore_index=True, sort=False)
    
    #Remove punctuation from all sentences
    dataframeReset = dataframe.reset_index(drop = False)
    for index, row in dataframeReset.iterrows():
        sentence = ''.join(ch for ch in row['CleanText'] if ch not in strng.punctuation)
        dataframeReset.loc[index,'CleanTextNoPunc'] = sentence
        
    dataframeNoStopReset = dataframeNoStop.reset_index(drop = False)
    for index, row in dataframeNoStopReset.iterrows():
        sentence = ''.join(ch for ch in row['CleanText'] if ch not in strng.punctuation)
        dataframeNoStopReset.loc[index,'CleanTextNoPunc'] = sentence
    
    
    # #########################################################################################
    #            Runs NaiveBayes and SVM per thread to see what performs best                 #
    # #########################################################################################
    if isRunPerThread:    
        #Create and assign the start of the return array the answer for the first accuracy score       
        accuracy_Array = []
        #initialize folds
        kf = KFold(n_splits = 3, shuffle = True, random_state = 45)
        goodSentences = dataframeReset[dataframeReset['FileName'].str.contains('summary')]
        summaryList = dataframeReset['CleanText'].isin(goodSentences['CleanText'])
        #The internet told me to split it like this
        for train_index, test_index in kf.split(dataframeReset, dataframeReset):
            #Create a new naive_bayes model for each test set and then put its accuracy in an array
            nb = MultinomialNB()            
            vect = TfidfVectorizer(ngram_range=(1, 2))
            
            # fit and transform training into vector matrix
            emails_train_dtm = vect.fit_transform(dataframeReset['CleanText'].iloc[train_index].values)
            emails_test_dtm = vect.transform(dataframeReset['CleanText'].iloc[test_index].values)
            
            #Fit and then compare the predictions
            nb.fit(emails_train_dtm, summaryList.iloc[train_index].values)
            category_prediction_test = nb.predict(emails_test_dtm)
            
            accuracy_Array.append(metrics.f1_score(summaryList.iloc[test_index].values, category_prediction_test))
                
        if len(accuracy_Array) > 0:
            naiveStatsArray.append({'ThreadName': folderName, 'F1_Score': sum(accuracy_Array)/float(len(accuracy_Array)), 'Confusion_Matrix': metrics.confusion_matrix(summaryList.iloc[test_index].values, category_prediction_test)})
           
        # ##########
        # SVM
        # ##########
            
        threads = []
        for randomState in range(1, 7):
            for cAmount in np.linspace(1, 100, 100):
                for gammaAmount in np.linspace(.1, .1, 1):  
                    threads.append(Thread(target=LearningThread, args=(dataframeReset[['FirstSentence', 'SecondSentence', 'ThirdSentence', 'FourthSentence', 'FifthSentence',
            'TopOneSentence', 'TopTwoSentence', 'TopThreeSentence', 'TopFourSentence', 'TopFiveSentence','SentenceLengthBeforeStop', 'CosineSimilarity']], dataframeNoStopReset[['FirstSentence', 'SecondSentence', 'ThirdSentence', 'FourthSentence', 'FifthSentence',
            'TopOneSentence', 'TopTwoSentence', 'TopThreeSentence', 'TopFourSentence', 'TopFiveSentence','SentenceLengthBeforeStop', 'CosineSimilarity']], summaryList, randomState, cAmount, gammaAmount)))
                    threads[-1].start()
                    
    return dataframeReset, dataframeNoStopReset
    
    
# Split the dataframes into the final data frame that will be used
# This includes matching up the summary sentences, vectorizing, and tfidf as well as assigning 
def ModifyRawData(cleanedDataFrame, cleanedEmails, cleanedDataSummaries, rawDataFrame, rawEmails, rawSummaries):
    # Create Y column (this is what we will be working to get using the SVM later on).  It's the unknown we want to solve for later on    
    rawEmails.reset_index(drop = False) 
    cleanedEmails.reset_index(drop = False)
    summaryList = rawEmails['CleanText'].isin(rawSummaries['CleanText'])
    cleanedSummaryList = cleanedEmails['CleanText'].isin(cleanedDataSummaries['CleanText'])
    
    if includeTFIDF:
        
        #Assign Test and Train parts
        rawEmailsNoPunc = rawEmails['CleanTextNoPunc']
        cleanedEmailsNoPunc = cleanedEmails['CleanTextNoPunc']
        
        rawtfidfVect = TfidfVectorizer(ngram_range=(1, 2))
        cleantfidfVect = TfidfVectorizer(ngram_range=(1, 2))
        
        #rawVect = CountVectorizer(ngram_range=(1, 2))
        #cleanVect = CountVectorizer(ngram_range=(1, 2))
        #hashVect = HashingVectorizer(ngram_range=(1, 2))
        
        
        #Create Series that the countVectorizer can use to create training and testing sets
        # ### Note: fits better at the moment without hashVect commented out in case of need for later.
        
        rawtfidfVect.fit(rawEmails['CleanTextNoPunc'])
        cleantfidfVect.fit(cleanedEmails['CleanTextNoPunc'])
        #rawVect.fit(rawEmails['CleanTextNoPunc'])
        #cleanVect.fit(rawEmails['CleanTextNoPunc'])
        #hashVect.fit(rawEmails)
            
        tfid_rawEmails_dtm = rawtfidfVect.transform(rawEmailsNoPunc)
        tfid_cleanEmails_dtm = cleantfidfVect.transform(cleanedEmailsNoPunc)
        # fit and transform training into vector matrix
        #vect_rawEmails_dtm = rawVect.transform(rawEmailsNoPunc)
        #vect_cleanEmails_dtm = rawVect.transform(cleanedEmailsNoPunc)
        #hash_rawEmails_train_dtm = hashVect.transform(rawEmails_train)
        
        # transform test into test matrix
        #vect_rawEmails_dtm = rawVect.transform(rawEmails_test)
        #tfid_rawEmails_dtm = rawtfidfVect.transform(rawEmails_test)
        #vect_clearnEmails_dtm = rawVect.transform(rawEmails_test)
        #tfid_cleanEmails_dtm = rawtfidfVect.transform(rawEmails_test)
        #hash_rawEmails_test_dtm = hashVect.transform(rawEmails_test)
        
        # #####################################################
        #               Cosine Similarity
        # #####################################################
        
        rawEmails = rawEmails.reset_index(drop = False)
        cleanedEmails = cleanedEmails.reset_index(drop = False)
        
        for folder in queryTFIDF:
            rawVector = rawtfidfVect.transform(np.array([re.sub('_', r' ', folder)]))
            cleanVector = cleantfidfVect.transform(np.array([re.sub('_', r' ', folder)]))
            
            # Raw Emails
            folderCosineComparisonFinished = False
            for index, row in rawEmails.iterrows():
                
                # loops until it finds first fileName that starts with folder name and then it restarts once it's done with all fileNames of that type
                if row['FileName'].startswith(folder) and not folderCosineComparisonFinished:
                    folderCosineComparisonFinished = True
                elif not row['FileName'].startswith(folder) and folderCosineComparisonFinished: 
                    continue
                
                if folderCosineComparisonFinished:
                    cosineSim = metrics.pairwise.cosine_similarity(tfid_rawEmails_dtm[index], rawVector)[0][0]
                    if cosineSim != 0:
                        rawEmails.loc[index, 'CosineSimilarity'] = cosineSim
                        
            # Cleaned Emails
            folderCosineComparisonFinished = False
            for index, row in cleanedEmails.iterrows():
                
                # loops until it finds first fileName that starts with folder name and then it restarts once it's done with all fileNames of that type
                if row['FileName'].startswith(folder) and not folderCosineComparisonFinished:
                    folderCosineComparisonFinished = True
                elif not row['FileName'].startswith(folder) and folderCosineComparisonFinished: 
                    continue
                
                if folderCosineComparisonFinished:
                    cosineSim = metrics.pairwise.cosine_similarity(tfid_cleanEmails_dtm[index], cleanVector)[0][0]
                    if cosineSim != 0:
                        cleanedEmails.loc[index, 'CosineSimilarity'] = cosineSim
            
        
        maxVal = max(rawEmails['CosineSimilarity'])
        for index, row in rawEmails.iterrows():
            rawEmails.loc[index, 'CosineSimilarity'] = row['CosineSimilarity'] / float(maxVal)
		
        maxVal = max(cleanedEmails['CosineSimilarity'])
        for index, row in cleanedEmails.iterrows():
            cleanedEmails.loc[index, 'CosineSimilarity'] = row['CosineSimilarity'] / float(maxVal)
        # #####################################################
        #               Combine TFIDF and CountVectorizer
        # #####################################################
        # Concatonate the columns of the training and test set
        #rawEmails_dtm = hstack([vect_rawEmails_dtm, tfid_rawEmails_dtm])
        #cleanEmails_dtm = hstack([vect_cleanEmails_dtm, tfid_cleanEmails_dtm])    
        #rawEmails_train_dtm = hstack([vect_tfidf_rawEmails_train_dtm, hash_rawEmails_train_dtm])
        #rawEmails_test_dtm = hstack([vect_tfidf_rawEmails_test_dtm, hash_rawEmails_test_dtm])
        
        
        
        rawEmails_dtm = hstack([tfid_rawEmails_dtm, rawEmails[['FirstSentence', 'SecondSentence', 'ThirdSentence', 'FourthSentence', 'FifthSentence',
            'TopOneSentence', 'TopTwoSentence', 'TopThreeSentence', 'TopFourSentence', 'TopFiveSentence','SentenceLengthBeforeStop', 'CosineSimilarity']].astype(float)])
        cleanEmails_dtm = hstack([tfid_cleanEmails_dtm, cleanedEmails[['FirstSentence', 'SecondSentence', 'ThirdSentence', 'FourthSentence', 'FifthSentence',
            'TopOneSentence', 'TopTwoSentence', 'TopThreeSentence', 'TopFourSentence', 'TopFiveSentence','SentenceLengthBeforeStop', 'CosineSimilarity']].astype(float)])
    else:
        rawEmails_dtm = rawEmails[['FirstSentence', 'SecondSentence', 'ThirdSentence', 'FourthSentence', 'FifthSentence',
            'TopOneSentence', 'TopTwoSentence', 'TopThreeSentence', 'TopFourSentence', 'TopFiveSentence','SentenceLengthBeforeStop', 'CosineSimilarity']].astype(float)
        cleanEmails_dtm = cleanedEmails[['FirstSentence', 'SecondSentence', 'ThirdSentence', 'FourthSentence', 'FifthSentence',
            'TopOneSentence', 'TopTwoSentence', 'TopThreeSentence', 'TopFourSentence', 'TopFiveSentence','SentenceLengthBeforeStop', 'CosineSimilarity']].astype(float)
    # This prints off indices of true values
    #print([i for i, x in enumerate(goodSentences_train) if x])
    return rawEmails_dtm, cleanEmails_dtm, summaryList

# #####################################################
#               Gets TFIDF for Naive Bayes
# #####################################################    
def GetTFIDF(cleanedDataFrame, cleanedEmails, cleanedDataSummaries, rawDataFrame, rawEmails, rawSummaries):
    # Create Y column (this is what we will be working to get using the SVM later on).  It's the unknown we want to solve for later on    
    rawEmails.reset_index(drop = False) 
    cleanedEmails.reset_index(drop = False)
    summaryList = rawEmails['CleanText'].isin(rawSummaries['CleanText'])
    cleanedSummaryList = cleanedEmails['CleanText'].isin(cleanedDataSummaries['CleanText'])
        
    #Assign Test and Train parts
    rawEmailsNoPunc = rawEmails['CleanTextNoPunc']
    cleanedEmailsNoPunc = cleanedEmails['CleanTextNoPunc']
    
    rawtfidfVect = TfidfVectorizer(ngram_range=(1, 2))
    cleantfidfVect = TfidfVectorizer(ngram_range=(1, 2))
    
    #Create Series that the countVectorizer can use to create training and testing sets
    rawtfidfVect.fit(rawEmails['CleanTextNoPunc'])
    cleantfidfVect.fit(cleanedEmails['CleanTextNoPunc'])
    tfid_rawEmails_dtm = rawtfidfVect.transform(rawEmailsNoPunc)
    tfid_cleanEmails_dtm = cleantfidfVect.transform(cleanedEmailsNoPunc)
        
    # This prints off indices of true values
    #print([i for i, x in enumerate(goodSentences_train) if x])
    return tfid_rawEmails_dtm, tfid_cleanEmails_dtm, summaryList   
    
    
    
    
#This should hold all the machine learning things
def MachineLearningPart(isSingleRun, rawEmails_dtm, cleanEmails_dtm, goodSentences):       
    
    # #########################################################################################
    #                       Simplest SVM Set up.  Used for one off runs                       #
    # #########################################################################################
    threads = []
    if isSingleRun:
        rawEmails_train, rawEmails_test, goodSentences_train, goodSentences_test = train_test_split(rawEmails_dtm, goodSentences, random_state=10)
        clf = svm.SVC(C=7, cache_size=1000, class_weight=None, coef0=0.1,
        decision_function_shape='ovr', degree=3, gamma=0.121606061, kernel='rbf',
        max_iter=-1, probability=False, random_state=1, shrinking=True,
        tol=.01, verbose=False)
        
        clf.fit(rawEmails_train, goodSentences_train)    
        
        vect_tfidf_emails_results = clf.predict(rawEmails_test)
        #print the accuracy is
        print('CountVectorizer + TFIDFVectorizer Results: ')
        
        #locStatsArray = pd.DataFrame.from_dict({'Sentences': rawEmails_test, 'ActualAmount': goodSentences_test, 'PredictedAmount': vect_tfidf_emails_results, 'F1_Score': metrics.f1_score(goodSentences_test, vect_tfidf_emails_results)})
        #locStatsArray.to_csv('Output/Sentence/NoTFIDF_7_121_10.csv', encoding='utf-8', index=False)
    
    # #########################################################################################
    #                                     Multi threaded                                      #
    # #########################################################################################
        
    else:
        for randomState in range(1, 7):
            for cAmount in np.linspace(1, 100, 100):
                    for gammaAmount in np.linspace(.1, .1, 1):  
                        threads.append(Thread(target=LearningThread, args=(rawEmails_dtm, cleanEmails_dtm, goodSentences, randomState, cAmount, gammaAmount)))
                        threads[-1].start()
    
    # '''
    # #########################################################################################
    #                                        NaiveBayes                                       #
    # #########################################################################################
    
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
    kf = KFold(n_splits = 3, shuffle = True, random_state = 45)
    
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
    
    
    # #########################################################################################
    #                           Coin flip and ZeroR for baselines                             #
    # ######################################################################################### 
    statsLock.acquire()
    statsArray.append({'Learning_Type': 'ZeroR F_1', 'cAmount': 0, 'gammaAmount': 0, 'randState': 0, 'F1_Score': metrics.f1_score(goodSentences, np.full((len(goodSentences), 1), False, dtype='bool')), 'Confusion_Matrix': metrics.confusion_matrix(goodSentences, np.full((len(goodSentences), 1), False, dtype='bool'))})
    statsLock.release()
    
    random.seed(123)
    for i in range(0, 10):
        coinTossArray = np.random.rand(len(goodSentences),1)
        coinTossArrayBool = (coinTossArray > .5)
        statsLock.acquire()
        statsArray.append({'Learning_Type': 'Coin Toss', 'cAmount': 0, 'gammaAmount': 0, 'randState': 0, 'F1_Score': metrics.f1_score(goodSentences, coinTossArrayBool), 'Confusion_Matrix': metrics.confusion_matrix(goodSentences, coinTossArrayBool)})
        statsLock.release()
    
    
    for thread in threads:
        """
        Waits for threads to complete before moving on with the main
        script.
        """
        thread.join()
    
    
def LearningThread(rawEmails_dtm, cleanEmails_dtm, goodSentences, randomState, cAmount, gammaAmount):

    rawEmails_train, rawEmails_test, goodSentences_train, goodSentences_test = train_test_split(rawEmails_dtm, goodSentences, random_state=randomState)
    
    
    # ###########################################################
    #                       Raw Emails SVM                      #
    # ########################################################### 
    
    clf = svm.SVC(C=cAmount, cache_size=5000, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=gammaAmount, kernel='rbf',
    max_iter=-1, probability=False, random_state=randomState, shrinking=True,
    tol=.001, verbose=False)

    clf.fit(rawEmails_train, goodSentences_train)    
    
    emails_train_dtm_results = clf.predict(rawEmails_test)
    
    # Only take "good SVMS" based on f1 score
    # Update to save all for CV comparison
    if (metrics.f1_score(goodSentences_test, emails_train_dtm_results) > 0):
        statsLock.acquire()
        statsArray.append({'Learning_Type': 'raw_SVM', 'cAmount': cAmount, 'gammaAmount': gammaAmount, 'randState': randomState, 'F1_Score': metrics.f1_score(goodSentences_test, emails_train_dtm_results), 'Confusion_Matrix': metrics.confusion_matrix(goodSentences_test, emails_train_dtm_results)})
        statsLock.release()
    
    
    # ###########################################################
    #                       Clean Emails SVM                    #
    # ###########################################################
    if not isRunPerThread:
        cleanedEmails_train, cleanedEmails_test, cleanGoodSentences_train, cleanGoodSentences_test = train_test_split(cleanEmails_dtm, goodSentences, random_state=randomState)
        clf = svm.SVC(C=cAmount, cache_size=5000, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=gammaAmount, kernel='rbf',
        max_iter=-1, probability=False, random_state=randomState, shrinking=True,
        tol=.001, verbose=False)

        clf.fit(cleanedEmails_train, cleanGoodSentences_train)    
        
        emails_train_dtm_results = clf.predict(cleanedEmails_test)
        
        # Only take "good SVMS" based on f1 score
        # Update to save all for CV comparison
        if (metrics.f1_score(cleanGoodSentences_test, emails_train_dtm_results) > 0):
            statsLock.acquire()
            statsArray.append({'Learning_Type': 'cleaned_SVM', 'cAmount': cAmount, 'gammaAmount': gammaAmount, 'randState': randomState, 'F1_Score': metrics.f1_score(cleanGoodSentences_test, emails_train_dtm_results), 'Confusion_Matrix': metrics.confusion_matrix(cleanGoodSentences_test, emails_train_dtm_results)})
            statsLock.release()
    
def main():    
   
    start_time = time.time()
    print(time.ctime(int(start_time)))
    
    # set up runType
    # Used for testing svm runs
    if len(sys.argv) >= 2 and sys.argv[1].upper() == 'SINGLE':
        isSingleRun = True
    else:
        isSingleRun = False
    
    # Decides whether to include TFIDF in SVM
    global includeTFIDF   
    if len(sys.argv) >= 3 and sys.argv[2].upper() == 'I':
        includeTFIDF = True
    else:
        includeTFIDF = False
        
    # machine learns based on email thread vs using all threads as corpus
    global isRunPerThread 
    if len(sys.argv) >= 4 and sys.argv[3].upper() == 'T':
        isRunPerThread = True
    else:
        isRunPerThread = False
        
    
    # ###########################################################
    #            Read in Files and create rawDataFrame          #
    # ###########################################################
        
    filepath = 'Sigcse/'
    df = pd.DataFrame(columns=['FileName', 'CleanText', 'CleanTextNoPunc'])
    dfNoStop = pd.DataFrame(columns=['FileName', 'TextNoStop', 'TextNoStopNoPunc'])
    dfTemp = pd.DataFrame(columns=['FileName', 'CleanText', 'CleanTextNoPunc'])
    dfNoStopTemp = pd.DataFrame(columns=['FileName', 'TextNoStop', 'TextNoStopNoPunc'])
    filenames= os.listdir(filepath)
    
    for filename in filenames: # loop through all the files and folders
    
        if os.path.isdir(os.path.join(os.path.abspath(filepath), filename)): # check whether the current object is a folder or not    
            # if empty insert into the new dataframe
            if df.empty:
                df, dfNoStop = LoopThroughDocuments(os.path.join(os.path.abspath(filepath), filename), filename)
            else:
                dfTemp, dfNoStopTemp = LoopThroughDocuments(os.path.join(os.path.abspath(filepath), filename), filename)
                df = pd.concat([df, dfTemp], ignore_index=True, sort=False)
                dfNoStop = pd.concat([df, dfNoStopTemp], ignore_index=True, sort=False)
    
    
    # #################################################################
    #            Modify Raw data into modified data vector            #
    #                     (will be normalized later)                  #
    # #################################################################    
    
    end_time = time.time()
    print(isRunPerThread)
    if not isRunPerThread:
        rawEmails_dtm, cleanEmails_dtm, goodSentences = ModifyRawData(dfNoStop, dfNoStop[dfNoStop['FileName'].str.contains('summary')==False], dfNoStop[dfNoStop['FileName'].str.contains('summary')], 
                            df,  df[df['FileName'].str.contains('summary')==False], df[df['FileName'].str.contains('summary')])
                            
                            
        MachineLearningPart(isSingleRun, rawEmails_dtm, cleanEmails_dtm, goodSentences)
        # prints full head
        # pd.set_option('display.max_colwidth', -1)
        # print(df.head())  
    else:
        naiveDataFrame = pd.DataFrame.from_dict(naiveStatsArray)
        naiveDataFrame.to_csv('Output/NaiveBayes/output_' + str(end_time) + '.csv', encoding='utf-8', index=False)

    locStatsArray = pd.DataFrame.from_dict(statsArray)
    print(time.ctime(int(end_time)))
    print('Runtime is: ' + str(end_time - start_time) + ' seconds.')
    locStatsArray = locStatsArray.sort_values(['F1_Score', 'cAmount', 'gammaAmount', 'randState', 'Learning_Type'], ascending=[False, True, True, True, True])
    #locStatsArray = locStatsArray.groupby(['cAmount', 'gammaAmount', 'randState', 'Learning_Type'])
    #print(locStatsArray)
    if includeTFIDF and not isRunPerThread:
        locStatsArray.to_csv('Output/outputWithTFIDF_' + str(end_time) + '.csv', encoding='utf-8', index=False)
    elif not includeTFIDF and not isRunPerThread:
        locStatsArray.to_csv('Output/outputNoTFIDF_' + str(end_time) + '.csv', encoding='utf-8', index=False)
    elif not includeTFIDF:
        locStatsArray.to_csv('Output/SVM/outputNoTFIDF_' + str(end_time) + '.csv', encoding='utf-8', index=False)
    else:
        locStatsArray.to_csv('Output/SVM/outputWithTFIDF_' + str(end_time) + '.csv', encoding='utf-8', index=False)
       
main()