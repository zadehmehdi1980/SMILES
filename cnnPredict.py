import numpy as np
from keras.models import model_from_json

from utils import fingerprint_features, X_reshape

def load_model():

    # load json and create model
    json_file = open('trainedModels/cnn_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("trainedModels/cnn_model.h5")
    print("Loaded model from local space")
    
    return loaded_model


def prediction(user_input, radius=2, size=2048):

    user_input = np.array(fingerprint_features(user_input, radius=radius, size=size))
    
    user_input = user_input.reshape(1, -1)

    user_input = X_reshape(user_input)

    model = load_model()

    return np.where(model.predict(user_input) >= 0.5, 1, 0)


# Please, uncomment below lines it while running stand-alone ( not in Docker with Flask entry point) 
"""
if __name__ == '__main__':

    print('Deploy')
    
    user_input = 'Cc1ccc(/C=C2\C(=O)NC(=O)N(Cc3ccccc3Cl)C2=O)o1'

    print('Predicted label: ', prediction(user_input))

"""
