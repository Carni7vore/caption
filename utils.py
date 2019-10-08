from inception_v3 import  InceptionV3, preprocess_input
import numpy as np
from tqdm import tqdm,trange
import cv2
import keras
from keras.preprocessing import image
from keras.models import Model

units=512
max_length=20

def load_image(image_path):
    img= cv2.imread(image_path)
    img= cv2.resize(img,(299,299))
    img = (img * 1.0) / 255
    img = img - 0.5
    img = img * 2
    return img


def batch_generator(batch_size,  vocab_size, train_num, start, image_name_train,senti_train, cap_vector_train):
    input_imgs = []
    # start=0
    # print(start,'start')

    input_sentis = np.array(senti_train)
    input_sequences = np.array(cap_vector_train)

    # print(input_sequences.shape,"input_seq")
    # target = input_sequences[:, 1:]

    # target= np.expand_dims(target,axis=-1)
    # print(target.shape,"target")
    # state_c = np.zeros(shape=(train_num, 512))
    # state_h = np.zeros(shape=(train_num, 512))

    while True:
        for i in range(start, start+train_num, batch_size):
            input_imgs = []
            for j in range(batch_size):
                img = load_image(image_name_train[i+j])
                input_imgs.append(img)
            img_features =np.array(input_imgs)
            seqs = input_sequences[i:i + batch_size, 0:max_length]
            state_c = np.zeros([batch_size, units])
            state_h = np.zeros([batch_size, units])
            sentiment = input_sentis[i:i + batch_size]

            target_seq = input_sequences[i:i + batch_size, 1:]
            y = keras.utils.to_categorical(target_seq, vocab_size)
            # print(y.shape,"y-shape")
            yield ([img_features, seqs, sentiment, state_c, state_h], y)


def extract_feature_from_image(file_dir):

    img = image.load_img(file_dir, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    base_model = InceptionV3(include_top=True, weights=None)
    weights_path = 'data/image_net.h5'

    base_model.load_weights(weights_path)


    new_input = base_model.layers[0].input
    hidden_layer = base_model.get_layer('avg_pool').output

    image_model = Model(new_input, hidden_layer)



    return image_model.predict(x)

def preprocess_img(img_name_vec):
    img_names = []
    feature_matrices = []

    for img_name in tqdm(img_name_vec):
        feature= extract_feature_from_image(img_name)
        img_names.append(img_name)
        feature_matrices.append(feature)

    return {'names': img_names, "features": feature_matrices}

