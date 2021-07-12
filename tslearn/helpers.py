'''
Authors: Jeff Adrion, Jared Galloway
'''

from tslearn.imports import *

def splitInt16(int16):
    '''
    Take in a 16 bit integer, and return the top and bottom 8 bit integers

    Maybe not the most effecient? My best attempt based on my knowledge of python
    '''
    int16 = np.uint16(int16)
    bits = np.binary_repr(int16, 16)
    top = int(bits[:8], 2)
    bot = int(bits[8:], 2)
    return np.uint8(top), np.uint8(bot)


def GlueInt8(int8_t, int8_b):
    '''
    Take in 2 8-bit integers, and return the respective 16 bit integer created
    byt gluing the bit representations together

    Maybe not the most effecient? My best attempt based on my knowledge of python
    '''
    int8_t = np.uint8(int8_t)
    int8_b = np.uint8(int8_b)
    bits_a = np.binary_repr(int8_t, 8)
    bits_b = np.binary_repr(int8_b, 8)
    ret = int(bits_a + bits_b, 2)
    return np.uint16(ret)

#-------------------------------------------------------------------------------------------

def progress_bar(percent, barLen = 50):
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()

#-------------------------------------------------------------------------------------------

def assign_task(mpID, task_q, nProcs):
    c,i,nth_job=0,0,1
    while (i+1)*nProcs <= len(mpID):
        i+=1
    nP1=nProcs-(len(mpID)%nProcs)
    for j in range(nP1):
        task_q.put((mpID[c:c+i], nth_job))
        nth_job += 1
        c=c+i
    for j in range(nProcs-nP1):
        task_q.put((mpID[c:c+i+1], nth_job))
        nth_job += 1
        c=c+i+1

#-------------------------------------------------------------------------------------------

def create_procs(nProcs, task_q, result_q, params, worker):
    pids = []
    for _ in range(nProcs):
        p = mp.Process(target=worker, args=(task_q, result_q, params))
        p.daemon = True
        p.start()
        pids.append(p)

    return pids

#-------------------------------------------------------------------------------------------

def get_mae(x,y):
    '''
    Compute mean absolute error between predictions and targets

    float[],float[] -> float
    '''
    assert(len(x) == len(y))
    summ = 0.0
    length = len(x)
    for i in range(length):
        summ += abs(x[i] - y[i])
    return summ/length

#-------------------------------------------------------------------------------------------

def get_mse(x,y):
    '''
    Compute mean squared error between predictions and targets

    float[],float[] -> float
    '''

    assert(len(x) == len(y))
    summ = 0.0
    length = len(x)
    for i in range(length):
        summ += (x[i] - y[i])**2
    return summ/length

#-------------------------------------------------------------------------------------------

def train_model(ModelFuncPointer,
            ModelName,
            targetLabels,
            TrainGenerator,
            ValidationGenerator,
            TestGenerator,
            resultsFile=None,
            numEpochs=10,
            epochSteps=100,
            network=None,
            nProc = 1,
            gpuID = 0):


    # configure the GPU
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpuID)
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import Session
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    Session(config=config)

    x,y = TrainGenerator.__getitem__(0)

    # early stopping and saving the best weights
    callbacks_list = [
            EarlyStopping(
                monitor='val_loss',
                verbose=1,
                min_delta=0.01,
                patience=250),
            ModelCheckpoint(
                filepath=network[1],
                monitor='val_loss',
                save_best_only=True)
            ]

    # fit model
    model = ModelFuncPointer(x,y)
    if nProc > 1:
        history = model.fit(TrainGenerator,
            epochs=numEpochs,
            validation_data=ValidationGenerator,
            callbacks=callbacks_list,
            use_multiprocessing=True,
            max_queue_size=nProc,
            workers=nProc)
    else:
        history = model.fit(TrainGenerator,
            epochs=numEpochs,
            validation_data=ValidationGenerator,
            callbacks=callbacks_list,
            use_multiprocessing=False)

    # write the model to disc
    if(network != None):
        ##serialize model to JSON
        model_json = model.to_json()
        with open(network[0], "w") as json_file:
            json_file.write(model_json)

    # reload json and weights
    if(network != None):
        jsonFILE = open(network[0],"r")
        loadedModel = jsonFILE.read()
        jsonFILE.close()
        model=model_from_json(loadedModel)
        model.load_weights(network[1])
    else:
        print("Error: model and weights not loaded")
        sys.exit(1)

    # predict on the test set
    x,y = TestGenerator.__getitem__(0)
    predictions = model.predict(x)

    # write the training history
    history.history['loss'] = np.array(history.history['loss'])
    history.history['val_loss'] = np.array(history.history['val_loss'])
    history.history['predictions'] = np.array(predictions)
    history.history['Y_test'] = np.array(y)
    history.history['name'] = ModelName
    history.history['target_labels'] = targetLabels
    print("results written to: ",resultsFile)
    pickle.dump(history.history, open( resultsFile, "wb" ))

    return history

#-------------------------------------------------------------------------------------------

def plotResults(resultsFile,saveas):

    '''
    plotting code for testing a model on simulation.
    using the resulting pickle file on a training run (resultsFile).
    This function plots the results of the final test set predictions,
    as well as validation loss as a function of Epochs during training.

    '''

    plt.rc('font', family='serif', serif='Times')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)

    results = pickle.load(open( resultsFile , "rb" ))

    target_labels = results["target_labels"]
    nTargets = len(target_labels)

    fig,axes = plt.subplots(nTargets+1,1)
    plt.subplots_adjust(hspace=0.9)

    for i in range(nTargets):
        predictions = np.array([float(Y[i]) for Y in results["predictions"]])
        realValues = np.array([float(X[i]) for X in results["Y_test"]])

        r_2 = round((np.corrcoef(predictions,realValues)[0,1])**2,5)

        mae = round(get_mae(realValues,predictions),4)
        mse = round(get_mse(realValues,predictions),4)
        labels = "$R^{2} = $"+str(r_2)+"\n"+"$mae = $" + str(mae)+" | "+"$mse = $" + str(mse)

        axes[i].scatter(realValues,predictions,marker = "o", color = 'tab:red',s=5.0,alpha=0.5)

        lims = [
            np.min([axes[i].get_xlim(), axes[i].get_ylim()]),
            np.max([axes[i].get_xlim(), axes[i].get_ylim()]),
        ]
        axes[i].set_xlim(lims)
        axes[i].set_ylim(lims)
        axes[i].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        axes[i].set_title(results["name"]+"\n"+labels,fontsize=12)

    axes[nTargets].plot(results["loss"],label = "mae loss",color='tab:grey')
    axes[nTargets].plot(results["val_loss"], label= "mae validation loss",color='tab:red')
    axes[nTargets].legend(frameon = False,fontsize = 12)
    axes[nTargets].set_ylabel("mse")
    axes[nTargets].set_xlabel("training epoch")

    for i, target in enumerate(target_labels):
        axes[i].set_ylabel("{} [{} predictions]".format(target,len(predictions)))
        axes[i].set_xlabel("{} [{} true values]".format(target,len(realValues)))
        axes[i].grid()

    width = 5.50
    height = (nTargets * 4.0) + 4.0
    fig.set_size_inches(width,height)
    fig.savefig(saveas)

#-------------------------------------------------------------------------------------------
