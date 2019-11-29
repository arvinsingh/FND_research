from FeatureGenerator import *
import ngram
import pickle
import pandas as pd
from time import time
from nltk.tokenize import sent_tokenize
from helpers import *
import hashlib
import nltk
import textstat


class ReadabilityFeatureGenerator(FeatureGenerator):
    """
    Readability is the ease with which a reader can understand
    a written text. In natural language, the readability of text
    depends on its content (the complexity of its vocabulary 
    and syntax) and its presentation
    """


    def __init__(self, name='readabilityFeatureGenerator'):
        super(ReadabilityFeatureGenerator, self).__init__(name)


    def process(self, df):

        t0 = time()
        print("\n---Generating Readability Features:---\n")

        def lexical_diversity(text):
            word_count = len(text)
            vocab_size = len(set(text))
            diversity_score = word_count / vocab_size
            return diversity_score

        def get_counts(text, word_list):
            words = nltk.tokenize.word_tokenize(text.lower())
            count = 0
            for word in words:
                if word in word_list:
                    count += 1
            return count

        df['flesch_reading_ease'] = df['articleBody'].astype(str).map(lambda x: textstat.flesch_reading_ease(x))
        print('flesch_reading_ease done!')
        df['smog_index'] = df['articleBody'].astype(str).map(lambda x: textstat.smog_index(x))
        print('smog_index done!')
        df['flesch_kincaid_grade'] = df['articleBody'].astype(str).map(lambda x: textstat.flesch_kincaid_grade(x))
        print('flesch_kincaid_grade done!')
        df['coleman_liau_index'] = df['articleBody'].astype(str).map(lambda x: textstat.coleman_liau_index(x))
        print('coleman_liau_index done!')
        df['automated_readability_index'] = df['articleBody'].astype(str).map(lambda x: textstat.automated_readability_index(x))
        print('automated_readability_index done!')
        df['dale_chall_readability_score'] = df['articleBody'].astype(str).map(lambda x: textstat.dale_chall_readability_score(x))
        print('dale_chall_readability_score done!')
        df['difficult_words'] = df['articleBody'].astype(str).map(lambda x: textstat.difficult_words(x))
        print('difficult_words done!')
        df['linsear_write_formula'] = df['articleBody'].astype(str).map(lambda x: textstat.linsear_write_formula(x))
        print('linsear_write_formula done!')
        df['gunning_fog'] = df['articleBody'].astype(str).map(lambda x: textstat.gunning_fog(x))
        print('gunning_fog done!')
        df['i_me_myself'] = df['articleBody'].astype(str).apply(get_counts,args = (['i', 'me', 'myself'],))
        print('i_me_myself done!')
        df['punct'] = df['articleBody'].astype(str).apply(get_counts,args = ([',','.', '!', '?'],))
        print('punct done!')
        df['lexical_diversity'] = df['articleBody'].astype(str).apply(lexical_diversity)
        print('lexical_diversity done!')

        feats = ['flesch_reading_ease', 'smog_index', 'flesch_kincaid_grade',
        'coleman_liau_index', 'automated_readability_index', 
        'dale_chall_readability_score', 'difficult_words', 'linsear_write_formula',
        'gunning_fog', 'i_me_myself', 'punct', 'lexical_diversity'
        ]


        outfilename_xReadable = df[feats].values

        with open('../saved_data/kaggle/read.pkl', 'wb') as outfile:
            pickle.dump(feats, outfile, -1)
            pickle.dump(outfilename_xReadable, outfile, -1)

        print ('readable features saved in read.pkl')
        
        print('\n---Readability Features is complete---')
        print("Time taken {} seconds\n".format(time() - t0))
        
        return 1


    def read(self):

        filename_rf = 'read.pkl'
        with open("../saved_data/new/" + filename_rf, "rb") as infile:
            _ = pickle.load(infile)
            xReadable = pickle.load(infile)
        print ('xReadable.shape: ', xReadable.shape)

        return [xReadable]

