
#read_sentiments.py

import numpy as np
# from gensim.models.word2vec import Word2Vec
# import gensim

# from inference_utils import vocabulary

fname= 'im2txt/stanfordSentimentTreebank/stanfordSentimentTreebank/sentiment_labels.txt'
fname2= 'im2txt/stanfordSentimentTreebank/stanfordSentimentTreebank/dictionary.txt'
fname3= 'im2txt/stanfordSentimentTreebank/stanfordSentimentTreebank/original_rt_snippets.txt'
fname4= 'im2txt/data/mscoco/word_counts.txt'

num= 239233
num3= 10605

def read_sentiments():
    dict_sentilabels = {}
    dict_phraseid = {}
    dict_sentiments = {}
    feature_size = 100
    sentences = []
    reverse_dict1=[]
    dict_words={}
    new_senti={}
    start_word = "<S>"
    end_word = "</S>"
    unk_word = "<UNK>"
    # vocab = vocabulary.Vocabulary("im2txt/data/mscoco/word_counts.txt")

    #dict from id to sentiments(floating num)
    with open(fname) as f:
        line=f.readline()
        for i in range(num-1):
            line= f.readline()
            line=line.strip()
            word = line.split('|')
            senti_label = word[1]
            phrase_id= word[0]
            dict_sentilabels[phrase_id]= float(senti_label)

    # dict from word/phrase to id
    # dict from word/phrase to sentiments(floating num)
    with open(fname2) as f2:
        for i in range(num-1):
            line=f2.readline()
            line=line.strip()
            word= line.split('|')
            phrase= word[0]
            id= word[1]
            dict_phraseid[phrase]=id
            dict_sentiments[phrase]= dict_sentilabels[id]

    # dict_sentiments['EOS']=0.499
    # dict_sentiments['UNK']=0.501

    # float word_senti/ int word senti
    with open(fname3) as f3:
        for i in range(num3):
            line=f3.readline()
            line= line.strip()
            # print(line)
            word = "".join((char if char.isalpha() else " ").lower() for char in line).split()
            # word.append("EOS")
            # word.append("UNK")
        # sentence=[]
            for w in word:
                j=0
                if w not in dict_words :
                    if w in dict_sentiments:
                        #dict_words[w]= int(5*dict_sentiments[w])
                        dict_words[w] =  5*dict_sentiments[w]

        #     if(len(sentences) <= 20):
        #         sentence.append(word)
        #     sentences.append(word)

    with open(fname4) as f4:
        reverse_vocab= list(f4.readlines())
        reverse_vocab = [line.split()[0] for line in reverse_vocab]
        reverse_dict1= reverse_vocab
        assert start_word in reverse_vocab
        assert end_word in reverse_vocab
        if unk_word not in reverse_vocab:
            reverse_vocab.append(unk_word)
        for i in range(len(reverse_vocab)):
            w= reverse_vocab[i]
            if w in dict_words:
                new_senti[i]= dict_words[w]
            else:
                new_senti[i]= int(3)
# bigram_transformer = gensim.models.Phrases(dict_sentiments.keys())
# m= Word2Vec(bigram_transformer[dict_sentiments.keys()],size=feature_size,window=5, min_count=1, workers=4)
#     m = Word2Vec(sentences, size=feature_size, window=5, min_count=3, workers=4)
    return new_senti, reverse_dict1
#




# dict_sentiments= read_sentiments()
# print (dict_sentiments['nice'])
# print (len(dict_sentiments))
# # # testword1= str('tan')
# testword1= "bicycle"
# # # print(dict_phraseid.keys())
# if testword1 in dict_sentiments:
#     print(testword1)
#     # print(dict_phraseid[testword1])
#     print(dict_sentiments[testword1])
#     word2= m1.wv[testword1]
#     # print(word2)
# else:
#     word2='unk'
#     print(word2)