'''
Authors: Jeff Adrion, Jared Galloway
'''

from tslearn.imports import *


def get_genome_coordinates(ts, dim=1):
    """
    Author: Jared Galloway
    Loop through the tree sequence and
    get all coordinates of each genome of diploid individual.

    :param dim: the dimentionality, 1,2, or 3.
    """

    coordinates = []
    for i in range(dim):
        coordinates.append(np.zeros(ts.num_samples))
    for ind in ts.individuals():
        for d in range(dim):
            for geno in range(2):
                coordinates[d][ind.nodes[geno]] = ind.location[d]

    return coordinates

#-------------------------------------------------------------------------------------------

def splitInt16(int16):
    '''
    Author: Jared Galloway
    Take in a 16 bit integer, and return the top and bottom 8 bit integers

    Maybe not the most effecient? My best attempt based on my knowledge of python
    '''
    int16 = np.uint16(int16)
    bits = np.binary_repr(int16, 16)
    top = int(bits[:8], 2)
    bot = int(bits[8:], 2)
    return np.uint8(top), np.uint8(bot)

#-------------------------------------------------------------------------------------------

def GlueInt8(int8_t, int8_b):
    '''
    Author: Jared Galloway
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

def weighted_trees(ts, sample_weight_list, node_fun=sum):
    '''
    Author: Jared Galloway
    Here ``sample_weight_list`` is a list of lists of weights, each of the same
    length as the samples in the tree sequence ``ts``. This returns an iterator
    over the trees in ``ts`` that is identical to ``ts.trees()`` except that
    each tree ``t`` has the additional method `t.node_weights()` which returns
    an iterator over the "weights" for each node in the tree, in the same order
    as ``t.nodes()``.

    Each node has one weight, computed separately for each set of weights in
    ``sample_weight_list``. Each such weight is defined for a particular list
    of ``sample_weights`` recursively:

    1. First define ``all_weights[ts.samples()[j]] = sample_weights[j]``
        and ``all_weights[k] = 0`` otherwise.
    2. The weight for a node ``j`` with children ``u1, u2, ..., un`` is
        ``node_fun([all_weights[j], weight[u1], ..., weight[un]])``.

    For instance, if ``sample_weights`` is a vector of all ``1``s, and
    ``node_fun`` is ``sum``, then the weight for each node in each tree
    is the number of samples below it, equivalent to ``t.num_samples(j)``.

    To do this, we need to only recurse upwards from the parent of each
    added or removed edge, updating the weights.
    '''
    samples = ts.samples()
    num_weights = len(sample_weight_list)
    # make sure the provided initial weights lists match the number of samples
    for swl in sample_weight_list:
        assert(len(swl) == len(samples))

    # initialize the weights
    base_X = [[0.0 for _ in range(num_weights)] for _ in range(ts.num_nodes)]
    X = [[0.0 for _ in range(num_weights)] for _ in range(ts.num_nodes)]
    # print(samples)
    for j, u in enumerate(samples):
        for k in range(num_weights):
            X[u][k] = sample_weight_list[k][j]
            base_X[u][k] = sample_weight_list[k][j]

    z = zip(ts.trees(tracked_samples=ts.samples()), ts.edge_diffs())
    for t, (interval, records_out, records_in) in z:
        for edge in itertools.chain(records_out, records_in):
            u = edge.parent
            while u != msprime.NULL_NODE:
                for k in range(num_weights):
                    U = None
                    if(t.is_sample(u)):
                        U = [base_X[u][k]] + [X[u][k] for u in t.children(u)]
                    else:
                        U = [X[u][k] for u in t.children(u)]
                    X[u][k] = node_fun(U)
                u = t.parent(u)

        def the_node_weights(self):
            for u in self.nodes():
                yield X[u]

        # magic that uses "descriptor protocol"
        t.node_weights = the_node_weights.__get__(t, msprime.SparseTree)
        # t.node_weights = the_node_weights(t)
        yield t

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

def mae(x,y):
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

def mse(x,y):
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
            TrainGenerator,
            ValidationGenerator,
            TestGenerator,
            resultsFile=None,
            numEpochs=10,
            epochSteps=100,
            validationSteps=1,
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
    #print(x)
    #print("x.shape:",x.shape)
    #print(y)
    #print("y.shape:",y.shape)
    #sys.exit()

    # early stopping and saving the best weights
    callbacks_list = [
            EarlyStopping(
                monitor='val_loss',
                verbose=1,
                min_delta=0.01,
                patience=20),
            ModelCheckpoint(
                filepath=network[1],
                monitor='val_loss',
                save_best_only=True)
            ]

    # fit model
    model = ModelFuncPointer(x,y)
    if nProc > 1:
        history = model.fit(TrainGenerator,
            steps_per_epoch=epochSteps,
            epochs=numEpochs,
            validation_data=ValidationGenerator,
            callbacks=callbacks_list,
            use_multiprocessing=True,
            max_queue_size=nProc,
            workers=nProc)
    else:
        history = model.fit(TrainGenerator,
            steps_per_epoch=epochSteps,
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
    print("results written to: ",resultsFile)
    pickle.dump(history.history, open( resultsFile, "wb" ))

    return None

#-------------------------------------------------------------------------------------------

def plotResults(resultsFile,saveas):

    '''
    plotting code for testing a model on simulation.
    using the resulting pickle file on a training run (resultsFile).
    This function plots the results of the final test set predictions,
    as well as validation loss as a function of Epochs during training.

    '''

    plt.rc('font', family='serif', serif='Times')
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)
    plt.rc('axes', labelsize=6)

    results = pickle.load(open( resultsFile , "rb" ))

    fig,axes = plt.subplots(3,1)
    plt.subplots_adjust(hspace=0.9)

    ##mu
    predictions = np.array([float(Y[0]) for Y in results["predictions"]])
    realValues = np.array([float(X[0]) for X in results["Y_test"]])

    r_2 = round((np.corrcoef(predictions,realValues)[0,1])**2,5)

    mae_0 = round(mae(realValues,predictions),4)
    mse_0 = round(mse(realValues,predictions),4)
    labels = "$R^{2} = $"+str(r_2)+"\n"+"$mae = $" + str(mae_0)+" | "+"$mse = $" + str(mse_0)

    axes[0].scatter(realValues,predictions,marker = "o", color = 'tab:purple',s=5.0,alpha=0.6)

    lims = [
        np.min([axes[0].get_xlim(), axes[0].get_ylim()]),  # min of both axes
        np.max([axes[0].get_xlim(), axes[0].get_ylim()]),  # max of both axes
    ]
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    axes[0].set_title(results["name"]+"\n"+labels,fontsize=6)

    ##rho
    predictions = np.array([float(Y[1]) for Y in results["predictions"]])
    realValues = np.array([float(X[1]) for X in results["Y_test"]])

    r_2 = round((np.corrcoef(predictions,realValues)[0,1])**2,5)

    mae_0 = round(mae(realValues,predictions),4)
    mse_0 = round(mse(realValues,predictions),4)
    labels = "$R^{2} = $"+str(r_2)+"\n"+"$mae = $" + str(mae_0)+" | "+"$mse = $" + str(mse_0)

    axes[1].scatter(realValues,predictions,marker = "o", color = 'tab:purple',s=5.0,alpha=0.6)

    lims = [
        np.min([axes[1].get_xlim(), axes[1].get_ylim()]),  # min of both axes
        np.max([axes[1].get_xlim(), axes[1].get_ylim()]),  # max of both axes
    ]
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)
    axes[1].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    axes[1].set_title(results["name"]+"\n"+labels,fontsize=6)

    axes[2].plot(results["loss"],label = "mae loss",color='tab:cyan')
    axes[2].plot(results["val_loss"], label= "mae validation loss",color='tab:pink')

    axes[2].legend(frameon = False,fontsize = 6)
    axes[2].set_ylabel("mse")

    axes[0].set_ylabel(str(len(predictions))+" mu predictions")
    axes[0].set_xlabel(str(len(realValues))+" mu real values")
    axes[1].set_ylabel(str(len(predictions))+" rho predictions")
    axes[1].set_xlabel(str(len(realValues))+" rho real values")
    fig.subplots_adjust(left=.15, bottom=.16, right=.85, top=.92,hspace = 0.5,wspace=0.4)
    height = 12.00
    width = 6.00

    axes[0].grid()
    axes[1].grid()
    fig.set_size_inches(width,height)
    fig.savefig(saveas)

#-------------------------------------------------------------------------------------------
