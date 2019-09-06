import tensorflow as tf
import json
import os
from dataset import Dataset
train_file="./train.txt"
test_file="./val.txt"
file_name="./train.txt"
dataset = Dataset(train_file, file_name)
print(dataset.test_size)
output_filename="ref_data.json"
test_length=20000

ref_data={}
ref_data['annotations']=[]

with open(file_name,"r") as f:
    for i in range(test_length):
        image, input, target, filename = dataset.next_batch(1, "test")
        sentence= "".join(target)
        image_id= "".join(filename)
        image_id = os.path.basename(image_id)
        image_id = image_id.replace("COCO_val2014_", "")
        print (image_id)
        image_id = image_id.replace(".jpg", "")
        image_id = "".join(image_id)
        image_id = int(image_id)
        ref_data['annotations'].append({u'image_id': image_id, u'caption': sentence})

print (ref_data['annotations'][0])
with open(output_filename,"w") as f:
  json.dump(ref_data,f)

with open("ref_out.txt","wb") as f:
    for item in ref_data['annotations']:
        f.write("%s \n"% item)
print("finished")