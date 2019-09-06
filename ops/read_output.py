import numpy as np
from ops import read_sentiments
#list words is a list that transfer number id to string words
#senti1 transfer number id to sentiments
senti1, list_words= read_sentiments.read_sentiments()
dict_words={}
for i in range(len(list_words)):
  dict_words[i]=list_words[i]
inv_dict= {v:k for k,v in dict_words.iteritems()}

def cap2num(cap_list):
  length1= len(cap_list)
  num1=[]
  words= cap_list.split(" ")
  for item in words:
    if item in inv_dict:
      word1= inv_dict[item]
      num1.append(str(word1))
  output= "".join(num1)
  return output

def num2cap(num_list):
  length1= len(num_list)
  cap1=[]
  for item in num_list:
    cap1.append(dict_words[item])

  return cap1



def read_output(fname):


  dict_sentence= {}
  with open(fname) as f:
    lines= list(f.readlines())
    l1= len(lines)

    for i in range(0,l1,2):
      w1 = lines[i].strip()
      # print w1
      w2 = lines[i+1].strip()
      w2= "".join((char if char.isalpha() else "") for char in w2)
      # print w2
      j=0
      while (w2.islower()):
        j+=1
        w1= w2
        w2= lines[i+j]
      w1 = [i.replace(".", "") for i in w1]
      w1 = [i.replace(",","") for i in w1]
      w1 = "".join(char for char in w1)
      # print(w1)
      #change to 0-4
      a= 2
      if w2=="Negative":
        a= 1
      elif w2=="Positive":
        a= 3
      elif w2=="Very negative":
        a= 0
      elif w2=="Very positive":
        a= 4
      elif w2=="Neutral":
        a= 2
      else:
        a= 2
      # print w1
      w1= w1.strip()
      # print w1
      w3= cap2num(w1)
      # print w3
      w3= "".join((char if char.isdigit() else "") for char in w3)

      dict_sentence[w3]= a
  return dict_sentence

# fname1= "output2.txt"
# dict1=read_output(fname1)
# print dict1.keys()[1]
# string1="a long restaurant table with rattan rounded back chairs"
# num1= cap2num(string1)
#
# print num1
#
# senti1= dict1[str(num1)]
# print(senti1)
#
# string2= str("a long restaurant table with rattan rounded back chairs")
# num2= cap2num(string2)
# print num2
# senti2= dict1[num2]
# print senti2
#
# string3= str("a man using a phone in a phone booth")
# num3= cap2num(string3)
# print num3
# senti3= dict1[num3]
# print senti3
#
# print dict1.values()[0:40]
#
