from tslearn.imports import *

def DENSE1(x,y):
    '''
    '''
    inputs = layers.Input(shape=(x.shape[1:]))
    model = layers.Flatten()(inputs)
    model = layers.BatchNormalization()(model)
    model = layers.Dense(512)(model)
    model = layers.Dense(512)(model)
    model = layers.Dropout(0.35)(model)
    model = layers.Dense(64)(model)
    model = layers.Dense(64)(model)
    model = layers.Dropout(0.35)(model)
    model = layers.Dense(2)(model)
    output = layers.Dense(2)(model)

    #----------------------------------------------------

    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer='Adam', loss='mse')
    model.summary()

    return model


def DENSE2(x,y):
    '''
    '''
    inputs = layers.Input(shape=(x.shape[1:]))
    model = layers.Flatten()(inputs)
    model = layers.BatchNormalization()(model)
    model = layers.Dense(128)(model)
    model = layers.Dropout(0.35)(model)
    model = layers.Dense(64)(model)
    model = layers.Dropout(0.35)(model)
    model = layers.Dense(2)(model)
    output = layers.Dense(2)(model)

    #----------------------------------------------------

    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer='Adam', loss='mse')
    model.summary()

    return model
