from tslearn.imports import *

def DENSE1(x,y):
    '''
    '''
    inputs = layers.Input(shape=(x.shape[1:]))
    model = layers.Flatten()(inputs)
    model = layers.BatchNormalization()(model)
    model = layers.Dense(128, activation='relu')(model)
    model = layers.Dropout(0.35)(model)
    model = layers.Dense(64, activation='relu')(model)
    model = layers.Dropout(0.35)(model)
    model = layers.Dense(y.shape[1])(model)
    output = layers.Dense(y.shape[1])(model)

    #----------------------------------------------------

    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer='Adam', loss='mse')
    model.summary()

    return model
