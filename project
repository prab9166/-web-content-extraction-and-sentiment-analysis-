import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
class DataIntiate():  # Initiates root variables and file paths

    def __init__(self, input_file=None, stop_file=None, pos_dict=None, neg_dict=None, country_code=None):
        try:
            # Reading input files
            self.data_input = pd.read_excel(input_file)  # Load main input file with URL_ID and URL
            self.data_stop = pd.read_csv(stop_file, header=None, names=['word'], encoding='ISO-8859-1')  # Load stopwords
            self.pos_words = pd.read_csv(pos_dict, header=None, names=['word'], encoding='ISO-8859-1')  # Load positive words
            self.neg_words = pd.read_csv(neg_dict, header=None, names=['word'], encoding='ISO-8859-1')  # Load negative words
            self.country_words = pd.read_csv(country_code, header=None, names=['word'], encoding='ISO-8859-1')  # Load country codes
            self.input_map = {'ID': 'URL_ID', 'url': 'URL'}  # Mapping column names

            # Initialize and preprocess data
            self.input_file()
            self.stop_file()
            self.pos_file()
            self.neg_file()
        except Exception as e:
            print("Error in initializing data:", e)

    def input_file(self):
        # Processes input file to extract ID and URL mappings
        try:
            result = {}
            for key, value in self.input_map.items():
                setattr(self, key, self.data_input.get(value))  # Dynamically set ID and url attributes

            id_list = getattr(self, 'ID')
            url_list = getattr(self, 'url')

            for i, (id_val, url_val) in enumerate(zip(id_list, url_list), 1):
                result[i] = f'{id_val} - {url_val}'  # Create mapping from ID to URL

            # Save the last pair to instance (could be changed to lists for multi-URL support)
            self.id_val = id_val
            self.url_val = url_val
        except Exception as e:
            print("Error in processing input file:", e)

    def stop_file(self):
        # Load stopwords list
        try:
            self.word = list(self.data_stop['word'])
            return self.word
        except Exception as e:
            print("Error in loading stop words:", e)

    def pos_file(self):
        # Load positive word list
        try:
            self.word = list(self.pos_words['word'])
            return self.word
        except Exception as e:
            print("Error in loading positive words:", e)

    def neg_file(self):
        # Load negative word list
        try:
            self.word = list(self.neg_words['word'])
            return self.word
        except Exception as e:
            print("Error in loading negative words:", e)

    def country_short(self):
        # Extract short country codes (e.g., from 'US - United States')
        try:
            self.codes = []
            for i in list(self.country_words['word']):
                code = i.split('-')[0].strip()
                self.codes.append(code)
            return self.codes
        except Exception as e:
            print("Error in processing country codes:", e)


class DataExt(DataIntiate):  # Extracts and analyzes content from a web page

    def data_crawl(self):
        try:
            self.url = r'https://insights.blackcoffer.com/callrail-analytics-leads-report-alert/'  # Static test URL
            req = requests.get(self.url)
            soup = BeautifulSoup(req.text, 'lxml')

            # Initialize counters for text stats
            self.phrase = []
            self.line_count = 0
            self.word_count = 0
            self.char_count = 0
            self.avg_sentencelen = 0
            self.vowels = {'a', 'e', 'i', 'o', 'u'}
            self.vowel_count = 0
            self.non_syllable = ('es', 'ed')
            self.syllable_count = 0
            self.complex_count = 0

            # Extract paragraph text
            outer_phrase = soup.find_all('div', class_='td-container')
            for inner_phrase in outer_phrase:
                heading = inner_phrase.find_all('strong')
                for lines in heading:
                    for texts in lines:
                        self.next_p = texts.find_next('p')
                        if self.next_p.text:
                            lines = self.next_p.text.strip().split('\n')
                            for line in lines:
                                if line.strip():
                                    self.line_count += 1
                                    for word in line.split(' '):
                                        if not word.lower().endswith(self.non_syllable):
                                            self.syllable_count += 1
                                            vowel_count_word = sum(1 for char in word.lower() if char in self.vowels)
                                            if vowel_count_word > 2:
                                                self.complex_count += 1

                                        self.word_count += 1
                                        self.char_count += len(word)
                                        self.vowel_count += sum(1 for char in word.lower() if char in self.vowels)

                            self.phrase.append(self.next_p.text.strip())

            # Derived stats calculations
            self.avg_word = self.char_count / self.word_count
            self.avg_sentencelen = self.word_count / self.line_count
            self.avg_complex_pcntg = self.complex_count / self.word_count
            self.fog_index = (0.4 * (self.avg_sentencelen + self.avg_complex_pcntg))

            # Summary output
            print(f'No.of lines : {self.line_count}\n')
            print(f'Total words from all lines : {self.word_count}\n')
            print(f'Total characters : {self.char_count}\n')
            print(f'Average word length is {self.avg_word:.2f}\n')
            print(f'Average sentence length: {self.avg_sentencelen:.2f}\n')
            print(f'Vowel counts : {self.vowel_count}\n')
            print(f'Syllable count : {self.syllable_count}\n')
            print(f'Complex word count : {self.complex_count}\n')
            print(f'Percentage of Complex words : {self.avg_complex_pcntg:.2f}\n')
            print(f'Fog index : {self.fog_index :.2f}\n')
        except Exception as e:
            print("Error in crawling or processing data:", e)


class DataAnalyse(DataExt):  # Perform sentiment and pronoun analysis

    def data_sementic(self):  # Typo in name (should be data_semantic)
        try:
            self.data_crawl()  # Process text extraction

            stopwords = self.stop_file()
            ps = PorterStemmer()
            corpus = []

            # Initialize sentiment variables
            self.pos_score = 0
            self.neg_score = 0
            self.pol_score = 0
            self.sub_score = 0
            self.avg_line = 0
            self.complex_score = 0
            self.personal_pro = 0
            self.avg_word = 0
            self.pronoun_count = 0
            pronouns = {"i", "we", "my", "ours", "us"}
            self.country_code = self.country_short()

            pos_words = set([ps.stem(pos_word.lower()) for pos_word in self.pos_file()])
            neg_words = set([ps.stem(neg_word.lower()) for neg_word in self.neg_file()])

            for i in self.phrase:
                review = re.sub('[^a-zA-Z0-1]', ' ', i)
                review = review.lower().split()
                review = [ps.stem(word) for word in review if not word in stopwords]
                corpus.append(review)

            self.len_corpus = sum(len(review) for review in corpus)
            print(f'Number of cleaned words :  {self.len_corpus}\n')

            for new_word in corpus:
                for text in new_word:
                    if text in pos_words:
                        self.pos_score += 1
                    if text in neg_words:
                        self.neg_score += 1
                    if text in pronouns and text not in self.country_code:
                        self.pronoun_count += 1

            self.pol_score = (self.pos_score - self.neg_score) / ((self.pos_score + self.neg_score) + 0.000001)
            self.sub_score = (self.pos_score - self.neg_score) / ((self.len_corpus) + 0.000001)

            print(f'Positive score is {self.pos_score}\n')
            print(f'Negative score is {self.neg_score}\n')
            print(f'Polarity score is {self.pol_score:.4f}\n')
            print(f'Subjectivity score is {self.sub_score:.4f}\n')
            print(f'Pronoun count : {self.pronoun_count}\n')
        except Exception as e:
            print("Error in semantic analysis:", e)


class MainFrame(DataAnalyse):  # Compile results and export to Excel
    def data_frame(self):
        try:
            self.data_sementic()

            # Prepare row for output
            data = list(zip([self.id_val], [self.url], [self.pos_score], [self.neg_score], [self.pol_score], [self.sub_score],
                            [self.avg_sentencelen], [self.avg_complex_pcntg], [self.fog_index], [self.line_count], [self.complex_count],
                            [self.word_count], [self.syllable_count], [self.pronoun_count], [self.len_corpus]))

            columns = ['URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH',
                       'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT',
                       'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']

            df = pd.DataFrame(data=data, columns=columns)

            # Export to Excel
            with pd.ExcelWriter(path=r"C:\Users\prabi\OneDrive\Desktop\project excels\final_output.xlsx", engine='xlsxwriter') as w:
                df.to_excel(w, index=False)

            print('Excel created\n')
        except Exception as e:
            print("Error in generating or writing DataFrame to Excel:", e)

# File paths for input data
file = r"D:\Files\20211030 Test Assignment\Input.xlsx"
stop = r"D:\Files\20211030 Test Assignment\StopWords\all_stopwords.txt"
pos = r"D:\Files\20211030 Test Assignment\MasterDictionary\positive-words.txt"
neg = r"D:\Files\20211030 Test Assignment\MasterDictionary\negative-words.txt"
country_txt = r"D:\Files\20211030 Test Assignment\StopWords\country_code.txt"

# Instantiate and run
try:
    output = MainFrame(input_file=file, stop_file=stop, pos_dict=pos, neg_dict=neg, country_code=country_txt)
    output.data_frame()
except Exception as e:
    print("Unexpected error during full execution:", e)
