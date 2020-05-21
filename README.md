# SentimentAnalysis Of Amazon product reviews

Dataset Information 

We use and compare various diffrent methods for sentiment analysis on Amazon Reviews(a binary classification problem).The training and testing datasets are .txt files having more than 25000 reviews combined.

Requirements
⦁	Numpy
⦁	Scikit-learn
⦁	scipy
⦁	nltk


Usage

Preprocessing and Training

1.	Run Untrained_Model.py for training of the raw data with multiple Classifiers.

2.  Since the training takes up too much time you can train once and pickle the trained classifiers.

3.  Once you pickle everythin needed you can run the trained model which will return the sentiment of input text.


Other Details:

For the Amazon web scraping script to run you need to have webdriver installed at path of your chrome.exe .
The default Path for some OS are as Follows:

Windows 10:

C:\Program Files (x86)\Google\Chrome\Application\chrome.exe

Windows 7:

C:\Program Files (x86)\Google\Application\chrome.exe

Vista:

C:\Users\UserName\AppDataLocal\Google\Chrome

XP:

C:\Documents and Settings\UserName\Local Settings\Application Data\Google\Chrome


Finally You run the Amazon_Reviews_predict.py file and it will scrape all the reviews and predict their seniment.


Information about other files
⦁	Train_and_Test_data contain about 10000 reviews of positive and negative sentiment each.
⦁	pickled_algos contain pickled trained models(Naive Bayes,LinearRegression...)


