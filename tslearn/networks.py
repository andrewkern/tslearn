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


def DENSE2(x,y):
    '''
    '''
    inputs0 = layers.Input(shape=(x[0].shape[1:]))
    model0 = layers.Flatten()(inputs0)
    model0 = layers.BatchNormalization()(model0)
    model0 = layers.Dense(16, activation='relu')(model0)
    model0 = layers.Dropout(0.35)(model0)

    inputs1 = layers.Input(shape=(x[1].shape[1:]))
    model1 = layers.Flatten()(inputs1)
    model1 = layers.BatchNormalization()(model1)
    model1 = layers.Dense(16, activation='relu')(model1)
    model1 = layers.Dropout(0.35)(model1)


    model = layers.concatenate([model0,model1])
    model = layers.Dense(16, activation='relu')(model)
    model = layers.Dropout(0.35)(model)
    model = layers.Dense(y.shape[1])(model)
    output = layers.Dense(y.shape[1])(model)

    #----------------------------------------------------

    model = Model(inputs=[inputs0,inputs1], outputs=[output])
    model.compile(optimizer='Adam', loss='mse')
    model.summary()

    return model


def DENSE4A(x,y):
    '''
    '''
    inputs0 = layers.Input(shape=(x[0].shape[1:]))
    #model0 = layers.Flatten()(inputs0)
    #model0 = layers.BatchNormalization()(model0)
    #model0 = layers.Dense(16, activation='relu')(model0)
    #model0 = layers.Dropout(0.35)(model0)

    inputs1 = layers.Input(shape=(x[1].shape[1:]))
    model1 = layers.Flatten()(inputs1)
    model1 = layers.BatchNormalization()(model1)
    model1 = layers.Dense(16, activation='relu')(model1)
    model1 = layers.Dropout(0.35)(model1)


    inputs2 = layers.Input(shape=(x[2].shape[1:]))
    #model2 = layers.BatchNormalization()(inputs2)
    #model2 = layers.Dense(16, activation='relu')(model2)
    #model2 = layers.Dropout(0.35)(model2)

    inputs3 = layers.Input(shape=(x[3].shape[1:]))
    model3 = layers.BatchNormalization()(inputs3)
    model3 = layers.Dense(16, activation='relu')(model3)
    model3 = layers.Dropout(0.35)(model3)


    #model = layers.concatenate([model0,model1,model2,model3])
    model = layers.concatenate([model1,model3])
    model = layers.Dense(16, activation='relu')(model)
    model = layers.Dropout(0.35)(model)
    model = layers.Dense(y.shape[1])(model)
    output = layers.Dense(y.shape[1])(model)

    #----------------------------------------------------

    model = Model(inputs=[inputs0,inputs1,
        inputs2, inputs3], outputs=[output])
    model.compile(optimizer='Adam', loss='mse')
    model.summary()

    return model


def DENSE4B(x,y):
    '''
    '''
    inputs0 = layers.Input(shape=(x[0].shape[1:]))
    #model0 = layers.Flatten()(inputs0)
    #model0 = layers.BatchNormalization()(model0)
    #model0 = layers.Dense(16, activation='relu')(model0)
    #model0 = layers.Dropout(0.35)(model0)

    inputs1 = layers.Input(shape=(x[1].shape[1:]))
    #model1 = layers.Bidirectional(layers.GRU(84,return_sequences=False))(inputs1)
    model1 = layers.Flatten()(inputs1)
    model1 = layers.BatchNormalization()(model1)
    model1 = layers.Dense(16, activation='relu')(model1)
    model1 = layers.Dropout(0.35)(model1)


    inputs2 = layers.Input(shape=(x[2].shape[1:]))
    #model2 = layers.BatchNormalization()(inputs2)
    #model2 = layers.Dense(16, activation='relu')(model2)
    #model2 = layers.Dropout(0.35)(model2)

    #inputs3 = layers.Input(shape=(x[3].shape[1:]))
    inputs3 = layers.Input(shape=(x[3].shape[1:]))
    model3 = layers.LSTM(4)(inputs3)
    #model3 = layers.BatchNormalization()(model3)
    model3 = layers.Dense(16, activation='relu')(model3)
    model3 = layers.Dropout(0.35)(model3)


    #model = layers.concatenate([model0,model1,model2,model3])
    model = layers.concatenate([model1,model3])
    model = layers.Dense(16, activation='relu')(model)
    model = layers.Dropout(0.35)(model)
    model = layers.Dense(y.shape[1])(model)
    output = layers.Dense(y.shape[1])(model)

    #----------------------------------------------------

    model = Model(inputs=[inputs0,inputs1,
        inputs2, inputs3], outputs=[output])
    model.compile(optimizer='Adam', loss='mse')
    model.summary()

    return model


def LSTM4A(x,y):
    '''
    '''
    inputs0 = layers.Input(shape=(x[0].shape[1:]))
    model0 = layers.LSTM(64)(inputs0)
    #model0 = layers.BatchNormalization()(model0)
    model0 = layers.Dense(64, activation='relu')(model0)
    model0 = layers.Dropout(0.35)(model0)

    inputs1 = layers.Input(shape=(x[1].shape[1:]))
    #model1 = layers.Bidirectional(layers.GRU(84))(inputs1)
    model1 = layers.LSTM(64)(inputs1)
    #model1 = layers.BatchNormalization()(model1)
    model1 = layers.Dense(64, activation='relu')(model1)
    model1 = layers.Dropout(0.35)(model1)


    inputs2 = layers.Input(shape=(x[2].shape[1:]))
    model2 = layers.LSTM(64)(inputs2)
    #model2 = layers.BatchNormalization()(inputs2)
    model2 = layers.Dense(64, activation='relu')(model2)
    model2 = layers.Dropout(0.35)(model2)

    inputs3 = layers.Input(shape=(x[3].shape[1:]))
    #model3 = layers.Bidirectional(layers.GRU(84))(inputs3)
    model3 = layers.LSTM(64)(inputs3)
    #model3 = layers.BatchNormalization()(model3)
    model3 = layers.Dense(64, activation='relu')(model3)
    model3 = layers.Dropout(0.35)(model3)


    #model = layers.concatenate([model0,model1,model2,model3])
    model = layers.concatenate([model0,model1,model2,model3])
    model = layers.Dense(64, activation='relu')(model)
    model = layers.Dropout(0.35)(model)
    model = layers.Dense(y.shape[1])(model)
    output = layers.Dense(y.shape[1])(model)

    #----------------------------------------------------

    model = Model(inputs=[inputs0,inputs1,
        inputs2, inputs3], outputs=[output])
    model.compile(optimizer='Adam', loss='mse')
    model.summary()

    return model


def LSTM4B(x,y):
    '''
    '''
    inputs0 = layers.Input(shape=(x[0].shape[1:]))
    model0 = layers.LSTM(64)(inputs0)
    #model0 = layers.BatchNormalization()(model0)
    model0 = layers.Dense(64, activation='relu')(model0)
    model0 = layers.Dropout(0.35)(model0)

    inputs1 = layers.Input(shape=(x[1].shape[1:]))
    #model1 = layers.Bidirectional(layers.GRU(84))(inputs1)
    model1 = layers.LSTM(64)(inputs1)
    #model1 = layers.BatchNormalization()(model1)
    model1 = layers.Dense(64, activation='relu')(model1)
    model1 = layers.Dropout(0.35)(model1)


    inputs2 = layers.Input(shape=(x[2].shape[1:]))
    model2 = layers.LSTM(64)(inputs2)
    #model2 = layers.BatchNormalization()(inputs2)
    model2 = layers.Dense(64, activation='relu')(model2)
    model2 = layers.Dropout(0.35)(model2)

    inputs3 = layers.Input(shape=(x[3].shape[1:]))
    #model3 = layers.Bidirectional(layers.GRU(84))(inputs3)
    model3 = layers.LSTM(64)(inputs3)
    #model3 = layers.BatchNormalization()(model3)
    model3 = layers.Dense(64, activation='relu')(model3)
    model3 = layers.Dropout(0.35)(model3)


    #model = layers.concatenate([model0,model1,model2,model3])
    model = layers.concatenate([model0,model1,model2,model3])
    model = layers.Dense(64, activation='relu')(model)
    model = layers.Dropout(0.35)(model)
    model = layers.Dense(y.shape[1])(model)
    output = layers.Dense(y.shape[1])(model)

    #----------------------------------------------------

    model = Model(inputs=[inputs0,inputs1,
        inputs2, inputs3], outputs=[output])
    model.compile(optimizer='Adam', loss='mse')
    model.summary()

    return model


def LSTM4C(x,y):
    '''
    '''
    inputs0 = layers.Input(shape=(x[0].shape[1:]))
    model0 = layers.LSTM(64)(inputs0)
    #model0 = layers.BatchNormalization()(model0)
    model0 = layers.Dense(128, activation='relu')(model0)
    model0 = layers.Dropout(0.35)(model0)

    inputs1 = layers.Input(shape=(x[1].shape[1:]))
    #model1 = layers.Bidirectional(layers.GRU(84))(inputs1)
    model1 = layers.LSTM(64)(inputs1)
    #model1 = layers.BatchNormalization()(model1)
    model1 = layers.Dense(128, activation='relu')(model1)
    model1 = layers.Dropout(0.35)(model1)


    inputs2 = layers.Input(shape=(x[2].shape[1:]))
    model2 = layers.LSTM(64)(inputs2)
    #model2 = layers.BatchNormalization()(inputs2)
    model2 = layers.Dense(128, activation='relu')(model2)
    model2 = layers.Dropout(0.35)(model2)

    inputs3 = layers.Input(shape=(x[3].shape[1:]))
    #model3 = layers.Bidirectional(layers.GRU(84))(inputs3)
    model3 = layers.LSTM(64)(inputs3)
    #model3 = layers.BatchNormalization()(model3)
    model3 = layers.Dense(128, activation='relu')(model3)
    model3 = layers.Dropout(0.35)(model3)


    #model = layers.concatenate([model0,model1,model2,model3])
    model = layers.concatenate([model0,model1,model2,model3])
    model = layers.Dense(256, activation='relu')(model)
    model = layers.Dropout(0.35)(model)
    model = layers.Dense(y.shape[1])(model)
    output = layers.Dense(y.shape[1])(model)

    #----------------------------------------------------

    model = Model(inputs=[inputs0,inputs1,
        inputs2, inputs3], outputs=[output])
    model.compile(optimizer='Adam', loss='mse')
    model.summary()

    return model


def LSTM4D(x,y):
    '''
    '''
    inputs0 = layers.Input(shape=(x[0].shape[1:]))
    model0 = layers.LSTM(64)(inputs0)
    model0 = layers.BatchNormalization()(model0)
    model0 = layers.Dense(128, activation='relu')(model0)
    model0 = layers.Dropout(0.35)(model0)

    inputs1 = layers.Input(shape=(x[1].shape[1:]))
    #model1 = layers.Bidirectional(layers.GRU(84))(inputs1)
    model1 = layers.LSTM(64)(inputs1)
    model1 = layers.BatchNormalization()(model1)
    model1 = layers.Dense(128, activation='relu')(model1)
    model1 = layers.Dropout(0.35)(model1)


    inputs2 = layers.Input(shape=(x[2].shape[1:]))
    model2 = layers.LSTM(64)(inputs2)
    model2 = layers.BatchNormalization()(model2)
    model2 = layers.Dense(128, activation='relu')(model2)
    model2 = layers.Dropout(0.35)(model2)

    inputs3 = layers.Input(shape=(x[3].shape[1:]))
    #model3 = layers.Bidirectional(layers.GRU(84))(inputs3)
    model3 = layers.LSTM(64)(inputs3)
    model3 = layers.BatchNormalization()(model3)
    model3 = layers.Dense(128, activation='relu')(model3)
    model3 = layers.Dropout(0.35)(model3)


    #model = layers.concatenate([model0,model1,model2,model3])
    model = layers.concatenate([model0,model1,model2,model3])
    model = layers.BatchNormalization()(model)
    model = layers.Dense(256, activation='relu')(model)
    model = layers.Dropout(0.35)(model)
    model = layers.Dense(y.shape[1])(model)
    output = layers.Dense(y.shape[1])(model)

    #----------------------------------------------------

    model = Model(inputs=[inputs0,inputs1,
        inputs2, inputs3], outputs=[output])
    model.compile(optimizer='Adam', loss='mse')
    model.summary()

    return model



def GRU4A(x,y):
    '''
    '''
    inputs0 = layers.Input(shape=(x[0].shape[1:]))
    #model0 = layers.LSTM(64)(inputs0)
    model0 = layers.Bidirectional(layers.GRU(64))(inputs0)
    #model0 = layers.BatchNormalization()(model0)
    model0 = layers.Dense(128, activation='relu')(model0)
    model0 = layers.Dropout(0.35)(model0)

    inputs1 = layers.Input(shape=(x[1].shape[1:]))
    #model1 = layers.Bidirectional(layers.GRU(84))(inputs1)
    #model1 = layers.LSTM(64)(inputs1)
    model1 = layers.Bidirectional(layers.GRU(64))(inputs1)
    #model1 = layers.BatchNormalization()(model1)
    model1 = layers.Dense(128, activation='relu')(model1)
    model1 = layers.Dropout(0.35)(model1)


    inputs2 = layers.Input(shape=(x[2].shape[1:]))
    #model2 = layers.LSTM(64)(inputs2)
    model2 = layers.Bidirectional(layers.GRU(64))(inputs2)
    #model2 = layers.BatchNormalization()(inputs2)
    model2 = layers.Dense(128, activation='relu')(model2)
    model2 = layers.Dropout(0.35)(model2)

    inputs3 = layers.Input(shape=(x[3].shape[1:]))
    #model3 = layers.Bidirectional(layers.GRU(84))(inputs3)
    #model3 = layers.LSTM(64)(inputs3)
    model3 = layers.Bidirectional(layers.GRU(64))(inputs3)
    #model3 = layers.BatchNormalization()(model3)
    model3 = layers.Dense(128, activation='relu')(model3)
    model3 = layers.Dropout(0.35)(model3)


    #model = layers.concatenate([model0,model1,model2,model3])
    model = layers.concatenate([model0,model1,model2,model3])
    model = layers.Dense(256, activation='relu')(model)
    model = layers.Dropout(0.35)(model)
    model = layers.Dense(y.shape[1])(model)
    output = layers.Dense(y.shape[1])(model)

    #----------------------------------------------------

    model = Model(inputs=[inputs0,inputs1,
        inputs2, inputs3], outputs=[output])
    model.compile(optimizer='Adam', loss='mse')
    model.summary()

    return model


def RELERNN_GRU(x,y):
    '''
    Same GPU used in ReLERNN
    '''

    haps,pos = x

    numSNPs = haps[0].shape[0]
    numSamps = haps[0].shape[1]
    numPos = pos[0].shape[0]

    genotype_inputs = layers.Input(shape=(numSNPs,numSamps))
    model = layers.Bidirectional(layers.GRU(84,return_sequences=False))(genotype_inputs)
    model = layers.Dense(256)(model)
    model = layers.Dropout(0.35)(model)

    #----------------------------------------------------

    position_inputs = layers.Input(shape=(numPos,))
    m2 = layers.Dense(256)(position_inputs)

    #----------------------------------------------------


    model =  layers.concatenate([model,m2])
    model = layers.Dense(64)(model)
    model = layers.Dropout(0.35)(model)
    output = layers.Dense(y.shape[1])(model)

    #----------------------------------------------------

    model = Model(inputs=[genotype_inputs,position_inputs], outputs=[output])
    model.compile(optimizer='Adam', loss='mse')
    model.summary()

    return model
