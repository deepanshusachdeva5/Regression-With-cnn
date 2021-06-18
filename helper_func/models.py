from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model


def create_mlp(dim, regress=False):
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation='relu'))
    model.add(Dense(4, activation='relu'))

    if regress:
        model.add(Dense(1, activation='linear'))

    return model


def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    inputShape = (height, width, depth)
    chanDim = -1

    # model input

    inputs = Input(shape=inputShape)

    for (i, f) in enumerate(filters):
        if i == 0:
            x = inputs

        x = Conv2D(f, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    x = Dense(4)(x)
    x = Activation('relu')(x)

    if regress:
        x = Dense(1, activation='linear')(x)

    model = Model(inputs, x)

    return model
