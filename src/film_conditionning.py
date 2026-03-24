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


def film_translate_spe(position, mods_code, mods_static, layers, activation):

    B = mods_code.shape[0]
    num_hidden = len(layers)

    dim = mods_code.shape[-1] // num_hidden
    mods_code = mods_code.view(B, num_hidden, dim)

    mods_static = mods_static.view(B, num_hidden, dim)

    h = position

    for i, l in enumerate(layers):

        beta  = mods_code[:, i, :]      # additif (code)
        gamma = mods_static[:, i, :]    # multiplicatif (static)

        h = l(h)


        h = (1 + gamma) * h + beta

        h = activation(h)

    return h
