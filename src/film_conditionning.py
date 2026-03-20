import torch 

def film_translate(position, mods, layers, activation):

    mods_batch = mods.shape[0]
    mods_nbr = mods.shape[-1]
    num_hidden = len(layers)


    mods = mods.reshape(mods_batch, num_hidden, mods_nbr // num_hidden)

    h = position



    for i, l in enumerate(layers):
        # CHECK SHAPE
        h = activation(l(h) + mods[:, i, :])


    return h

def film_translate2(position, features, layers, activation):

    feature_shape = features.shape[0]
    feature_dim = features.shape[-1]
    num_hidden = len(layers)

    features = features.reshape(feature_shape, 1, num_hidden, feature_dim // num_hidden)

    h = position

    for i, l in enumerate(layers):
        # CHECK SHAPE
        h = activation(l(h) + features[..., i, :])


    return h
