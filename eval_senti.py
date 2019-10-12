import matplotlib.pyplot as plt
import numpy as np
import json
fname2= "output/out_data_neu.txt"

fname3= "output/out_data_pos.txt"

fname1= "output/out_data_neg.txt"
distrib3=np.zeros((3,3))
with open(fname1) as f:
    lines = list(f.readlines())
    l1 = len(lines)
    for i in range(0,l1):

      w2 = lines[i].strip()
      j = 0

      if w2=="Negative":
        distrib3[0,0]+=1
      elif w2=="Positive":
        distrib3[0,2]+=1
      elif w2=="Very negative":
        distrib3[0,0]+=1
      elif w2=="Very positive":
        distrib3[0,2]+=1
      elif w2=="Neutral":
        distrib3[0,1]+=1


with open(fname2) as f:
    lines = list(f.readlines())
    l1 = len(lines)
    for i in range(0,l1):

      w2 = lines[i].strip()
      j = 0

      if w2=="Negative":
        distrib3[1,0]+=1
      elif w2=="Positive":
        distrib3[1,2]+=1
      elif w2=="Very negative":
        distrib3[1,0]+=1
      elif w2=="Very positive":
        distrib3[1,2]+=1
      elif w2=="Neutral":
        distrib3[1,1]+=1


with open(fname3) as f:
    lines = list(f.readlines())
    l1 = len(lines)
    for i in range(0,l1):

      w2 = lines[i].strip()
      j = 0

      if w2=="Negative":
        distrib3[2,0]+=1
      elif w2=="Positive":
        distrib3[2,2]+=1
      elif w2=="Very negative":
        distrib3[2,0]+=1
      elif w2=="Very positive":
        distrib3[2,2]+=1
      elif w2=="Neutral":
        distrib3[2,1]+=1


print(distrib3)
with open('output/confusion_mat.txt','w') as f2:
    for line in distrib3:
        f2.write(str(line)+'\n')