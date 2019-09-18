import numpy as np
import matplotlib.pyplot as plt
fname="out_train.txt"
i=0
l1=10000
senti_score=[]
senti_avg=[]
senti_var=[]
with open(fname) as f1:
	while(i<l1):
		w1= f1.readline().rstrip()
		w2= f1.readline().rstrip()

		w2= "".join((char if char.isalpha() else "") for char in w2)
		# print(w2)
		# while (w2.islower()):
		# 	i+=1
		# 	w1= w2
		# 	w2= f1.readline().rstrip()
		# 	w2= "".join((char if char.isalpha() else "") for char in w2)

		a= 0.5
		if w2=="Negative":
			a= 0.25
		elif w2=="Positive":
			a= 0.75
		elif w2=="Very negative":
			a= 0
		elif w2=="Very positive":
			a= 1
		elif w2=="Neutral":
			a= 0.5

		senti_score.append(a)
		i+=1
# print(senti_score[0:500] )

for i in range(0,l1,5):
	avg= sum(senti_score[i:i+5])
	variance= np.var(senti_score[i:i+5])
	senti_avg.append(avg)
	senti_var.append(variance)

plt.figure()
plt.plot(senti_avg,"o")
plt.title("mean")
plt.show()
plt.figure()
plt.plot(senti_var,"o")
plt.title("variance")
plt.show()

# print(senti_avg[0:500])
print(senti_var[0:500])