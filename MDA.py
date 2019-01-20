from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Dense, concatenate
from keras import regularizers


def build_AE(input_dim, encoding_dims):
    """
    Function for building autoencoder.
    """
    # input layer
    input_layer = Input(shape=(input_dim, ))
    hidden_layer = input_layer
    for i in range(0, len(encoding_dims)):
        # generate hidden layer
        if i == len(encoding_dims)/2:
            hidden_layer = Dense(encoding_dims[i],
                                 activation='sigmoid',
                                 # activity_regularizer=regularizers.l1(10e-6),
                                 name='middle_layer')(hidden_layer)
        else:
            hidden_layer = Dense(encoding_dims[i],
                                 activation='sigmoid',
                                 name='layer_' + str(i+1))(hidden_layer)

    # reconstruction of the input
    decoded = Dense(input_dim,
                    activation='sigmoid')(hidden_layer)

    # autoencoder model
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
    model = Model(inputs=input_layer, outputs=decoded)
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    print (model.summary())

    return model


def build_MDA(input_dims, encoding_dims):
    """
    Function for building multimodal autoencoder.
    """
    # input layers
    input_layers = []
    for dim in input_dims:
        input_layers.append(Input(shape=(dim, )))

    # hidden layers
    hidden_layers = []
    for j in range(0, len(input_dims)):
        hidden_layers.append(Dense(encoding_dims[0]/len(input_dims),
                                   # activity_regularizer=regularizers.l1(gamma[j]),
                                   activation='sigmoid')(input_layers[j]))

    # Concatenate layers
    if len(encoding_dims) == 1:
        hidden_layer = concatenate(hidden_layers, name='middle_layer')
    else:
        hidden_layer = concatenate(hidden_layers)

    # middle layers
    for i in range(1, len(encoding_dims)-1):
        if i == len(encoding_dims)/2:
            hidden_layer = Dense(encoding_dims[i],
                                 name='middle_layer',
                                 # kernel_regularizer=regularizers.l1(1e-5),
                                 activation='sigmoid')(hidden_layer)
        else:
            hidden_layer = Dense(encoding_dims[i],
                                 # kernel_regularizer=regularizers.l1(1e-5),
                                 activation='sigmoid')(hidden_layer)

    if len(encoding_dims) != 1:
        # reconstruction of the concatenated layer
        hidden_layer = Dense(encoding_dims[0],
                             activation='sigmoid')(hidden_layer)

    # hidden layers
    hidden_layers = []
    for j in range(0, len(input_dims)):
        hidden_layers.append(Dense(encoding_dims[-1]/len(input_dims),
                                   activation='sigmoid')(hidden_layer))
    # output layers
    output_layers = []
    for j in range(0, len(input_dims)):
        output_layers.append(Dense(input_dims[j],
                                   activation='sigmoid')(hidden_layers[j]))

    # autoencoder model
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
    model = Model(inputs=input_layers, outputs=output_layers)
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    print (model.summary())

    return model


def build_MDA2(input_dims, encoding_dims):
    """
    Function for building mixed multimodal autoencoder.
    """
    # input layers
    input_layers = []
    for dim in input_dims:
        input_layers.append(Input(shape=(dim, )))

    # ENCODER

    # hidden layer
    hidden_layers = input_layers
    for i in range(0, len(encoding_dims)-1):
        tmp_layers = []
        if isinstance(encoding_dims[i], list) and isinstance(encoding_dims[i+1], int):
            tmp1_layers = []
            for j in range(0, len(encoding_dims[i])):
                tmp1_layers.append(Dense(encoding_dims[i][j],
                                         # kernel_initializer='random_uniform',
                                         activation='sigmoid')(hidden_layers[j]))
            tmp_layers.append(concatenate(tmp1_layers))
        elif isinstance(encoding_dims[i], int):
            tmp_layers.append(Dense(encoding_dims[i],
                                    # kernel_initializer='random_uniform',
                                    activation='sigmoid')(hidden_layers[0]))
        else:
            for j in range(0, len(encoding_dims[i])):
                tmp_layers.append(Dense(encoding_dims[i][j],
                                        # kernel_initializer='random_uniform',
                                        # activity_regularizer=regularizers.l1(0.0001),
                                        activation='sigmoid')(hidden_layers[j]))
        hidden_layers = tmp_layers

    # middle layer
    tmp_layers = []
    tmp_layers.append(Dense(encoding_dims[-1],
                            name='middle_layer',
                            # kernel_initializer='random_uniform',
                            # activity_regularizer=regularizers.l1(10e-6),
                            activation='sigmoid')(hidden_layers[0]))
    hidden_layers = tmp_layers

    # DECODER

    # hidden layers
    for i in range(2, len(encoding_dims) + 1):
        tmp_layers = []
        if isinstance(encoding_dims[-i], int):
            tmp_layers.append(Dense(encoding_dims[-i],
                                    # kernel_initializer='random_uniform',
                                    activation='sigmoid')(hidden_layers[0]))
        elif isinstance(encoding_dims[-i], list) and isinstance(encoding_dims[-i+1], int):
            tmp = Dense(sum(encoding_dims[-i]),
                        # kernel_initializer='random_uniform',
                        activation='sigmoid')(hidden_layers[0])
            for j in range(0, len(encoding_dims[-i])):
                tmp_layers.append(Dense(encoding_dims[-i][j],
                                        # kernel_initializer='random_uniform',
                                        activation='sigmoid')(tmp))
        else:
            for j in range(0, len(encoding_dims[-i])):
                tmp_layers.append(Dense(encoding_dims[-i][j],
                                        # kernel_initializer='random_uniform',
                                        activation='sigmoid')(hidden_layers[j]))
        hidden_layers = tmp_layers

    # output layers
    output_layers = []
    for j in range(0, len(input_dims)):
        output_layers.append(Dense(input_dims[j],
                                   # kernel_initializer='random_uniform',
                                   activation='sigmoid')(hidden_layers[j]))

    # autoencoder model
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
    model = Model(inputs=input_layers, outputs=output_layers)
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    print (model.summary())

    return model
