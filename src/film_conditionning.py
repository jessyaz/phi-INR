import torch 

def film_translate(position, features, layers, activation):

    feature_shape = features.shape[0]
    feature_dim = features.shape[-1]
    num_hidden = len(layers)

    features = features.reshape(feature_shape, 1, num_hidden, feature_dim // num_hidden)

    h = position

    for i, l in enumerate(layers):
        # CHECK SHAPE
        h = activation(l(h) + features[..., i, :])


    return h

