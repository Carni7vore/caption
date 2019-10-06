from inception_v3 import  InceptionV3, preprocess_input
from keras.preprocessing import  image
from keras.models import Model
import os
import numpy as np


def extract_features(directory):
    base_model = InceptionV3(include_top=True, weights=None)
    weights_path = 'data/image_net.h5'

    base_model.load_weights(weights_path)

    new_input = base_model.layers[0].input
    hidden_layer = base_model.get_layer('avg_pool').output

    image_model = Model(new_input, hidden_layer)

    img_id = []
    img_matrices = []
    for img_file in os.listdir(directory):
        img_path = directory + '/' + img_file
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        img_id.append(os.path.splitext(img_file)[0])
        img_matrices.append(x)

    img_matrices = np.array(img_matrices)
    assert (len(img_matrices.shape) == 4)

    img_features = image_model.predict(img_matrices, verbose=1)

    return {'ids': img_id, 'features': img_features}

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

