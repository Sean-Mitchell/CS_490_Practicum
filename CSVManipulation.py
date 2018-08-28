import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold

# Read in all the CSV Files
def LoopThroughDocuments(counter, dataframe):
	#Made the encoding whatever ISO is because otherwise the encoding would break for some reason (dates??)
	df = pd.read_csv('D:\Documents\CS_490_Practicum\Sean_Report\SeanReport' + str(counter) + '.csv', encoding="ISO-8859-1")

	#Concatonate DataFrames to what's already existing
	dataframe = pd.concat([dataframe, df[['Issue Number','Brief Description','Category']]])
	
	#Prints the report being read in
	#print('Report ' + str(counter))
	return dataframe.reindex()
	
#This should hold all the machine learning things
def MachineLearningPart(dataframe):
	vect = CountVectorizer()
	
	#Assign Test and Train parts
	descriptionX = dataframe['Brief Description']
	categoryY = dataframe['Category']
	
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
	#Create catchAll df with 3 columns, Id, Description, Category
	concatdf = pd.DataFrame(columns=['Issue Number','Brief Description','Category'])
	
	
	#Reads all documents and displays them in log
	for x in range(1,13):
		concatdf = LoopThroughDocuments(x, concatdf)
	
	#Get before shape
	#print(concatdf.shape)
	
	#Drop NaN values 
	concatdf = concatdf.dropna()
	
	#Get Results (first in array is the accuracy from the split)
	#All else are accuracies from the kfold split
	nb_accuracy = MachineLearningPart(concatdf)
	
	
	print('Split Accuracy: ' + str(nb_accuracy[0]))
	print('Kfold Accuracy Ave: ' + str(sum(nb_accuracy[1:])/len(nb_accuracy[1:])))
	
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