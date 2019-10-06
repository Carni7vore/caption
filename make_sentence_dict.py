import json
dict_sentence= {}
dict_name="sentence_dict2.json"
error=0
for i in range(1):
    fp= "out_train.txt"
    with open(fp,"r") as fp:
        lines= fp.readlines()
        l1= len(lines)
        j=0
        while(j<l1-1):
            w0= lines[j].strip()
            w1= lines[j+1].strip()
            while (w1.islower()):
                j += 1
                w0 = w1
                w1 = lines[j+1].strip()
            w0 = [x.replace(".", "") for x in w0]
            w0 = [x.replace(",", "") for x in w0]
            w0 = [x.replace("'", "") for x in w0]
            w0 = [x.replace("?", "") for x in w0]
            w0= "".join(x for x in w0)
            w0= w0.rstrip()
            j+=2
            a = 0
            if w1 == "Negative":
                a = -1
            elif w1 == "Positive":
                a = 1
            elif w1 == "Very negative":
                a = -1
            elif w1 == "Very positive":
                a = 1
            elif w1 == "Neutral":
                a = 0
            else:
                error+=1
                print(w0)
                print(w1)
            dict_sentence[w0]=a
print(error)


with open(dict_name,"w") as f2:
    json.dump(dict_sentence,f2)