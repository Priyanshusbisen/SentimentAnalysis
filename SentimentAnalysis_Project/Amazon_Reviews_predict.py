from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import urllib.request
import pickle
from SentimentAnalysis_TrainedModel import sentiment
import sys



vote_list = []

class AmazonScraper():
    def __init__(self,number,url):
        self.number = number
        self.url = f'{url}&pageNumber={number}'
        self.driver = webdriver.Chrome('C:\\Program Files (x86)\\Google\\Chrome\\Application\\chromedriver')
        self.delay = 3

    def Load_url(self):
        self.driver.get(self.url)
        
        try:
            wait = WebDriverWait(self.driver,self.delay)
            wait.until(EC.presence_of_element_located((By.ID,'a-page')))
        except TimeoutException:
            print('Loading took too much time.')

    def scrape_data(self):
        
        all_comments  = self.driver.find_elements_by_class_name('a-section.review.aok-relative')
        for comment in all_comments:
            if comment.find_element_by_class_name('a-row.a-spacing-small.review-data'):
                comment = comment.find_element_by_class_name('a-row.a-spacing-small.review-data')
                senti = sentiment(comment.text)
                vote_list.append(senti)
                
    def final_review(self):
        pos_votes = vote_list.count('pos')
        neg_votes = vote_list.count('neg')
        print(f'The comment secion has {(pos_votes/len(vote_list))*100}% positive and {(neg_votes/len(vote_list))*100}% negative comments.')
        print(f'Number of postive comments = {pos_votes}')
        print(f'Number of negative comments = {neg_votes}')

    def Close_tab(self):
        self.driver.close()

    def num_comments(self):
        num = self.driver.find_element_by_id('filter-info-section')
        num = int(''.join(num.text.split(' ')[-2].split(',')))     
        return num
number = 1
url = input('Enter URL here:')

Data = AmazonScraper(number)
Data.Load_url()
Data.scrape_data()
total_num = Data.num_comments()//10
Data.Close_tab()

for i in range(2,total_num+1):
    Data = AmazonScraper(i)
    Data.Load_url()
    Data.scrape_data()
    Data.Close_tab()

Data.final_review()

        














