'''
Authors: Jared Galloway, Jeff Adrion
'''

from tslearn.imports import *
from tslearn.encoder import *

class treesBatchGenerator(tf.keras.utils.Sequence):

    '''
    This class, encodedTreesGenerator, extends tf.keras.utils.Sequence.
    So as to multithread the batch preparation in tandum with network training
    for maximum effeciency on the hardware provided.

    It generated batches of encoded tree sequences from a given .trees directory
    (which is generated most effeciently from the Simulator class)
    which have been prepped according to the given parameters.

    It also offers a range of data prepping heuristics as well as normalizing
    the targets.

    def __getitem__(self, idx):

    def __data_generation(self, batchTreeIndices):

    '''

    #Initialize the member variables which largely determine the data prepping heuristics
    #in addition to the .trees directory containing the data from which to generate the batches
    def __init__(self,
            treesDirectory,
            targetNormalization = 'binary',
            batchSize=64,
            frameWidth=0,
            center=False,
            shuffleExamples = False,
            maxTsTableSize = None,
            numReps = None,
            rawTargets = None,
            ReLERNN = None
            ):

        self.treesDirectory = treesDirectory
        self.targetNormalization = targetNormalization
        self.batch_size = batchSize
        self.frameWidth = frameWidth
        self.center = center
        self.shuffleExamples = shuffleExamples
        self.maxTsTableSize = maxTsTableSize
        self.numReps = numReps
        self.indices = np.arange(numReps)
        self.rawTargets = rawTargets
        infoFilename = os.path.join(self.treesDirectory,"info.p")
        self.infoDir = pickle.load(open(infoFilename,"rb"))
        self.ReLERNN = ReLERNN
        if(targetNormalization != None):
            self.normalizedTargets = self.normalizeTargets()
        if(shuffleExamples):
            np.random.shuffle(self.indices)


    def pad_encodings(self, encodings, maxTsTableSize=None, frameWidth=0, center=False):
        """
        pads all encodings to the max size of the tree sequence tables
        maxTsTableSize corresponds to the max lengths of of the
        [nodes, edges, sites, mutations, and trees] table
        """

        ### Just the mutation matrix
        #maxHeight = int(maxTsTableSize[2])
        #maxWidth = int(maxTsTableSize[0])
        #pad_val = 0.0
        #for i in range(len(encodings)):
        #    height = encodings[i].shape[0]
        #    width = encodings[i].shape[1]
        #    paddingHeight = maxHeight - height
        #    paddingWidth = maxWidth - width
        #    if(center):
        #        priorH = paddingHeight // 2
        #        postH = paddingHeight - priorH
        #        priorW = paddingWidth // 2
        #        postW = paddingWidth - priorW
        #        encodings[i] = np.pad(encodings[i],((priorH,postH),(priorW,postW),(0,0)),"constant",constant_values=pad_val)
        #    else:
        #        print("Error: currently center must be set to True")
        #        sys.exit(1)
        #encodings = np.array(encodings)

        #if(frameWidth):
        #    fw = frameWidth
        #    encodings = np.pad(encodings,((0,0),(fw,fw),(fw,fw),(fw,fw)),"constant",constant_values=pad_val)

        #return encodings



        #### mutation matrix and edgeXnodeXnode matrix
        ## normalize edge encodings
        #edge_encodings = []
        #maxHeight = int(maxTsTableSize[1])
        #maxWidth = int(maxTsTableSize[0]+2)
        #maxDepth = int(maxTsTableSize[0])
        #pad_val = 0.0
        #for i in range(len(encodings)):
        #    height = encodings[i][0].shape[0]
        #    width = encodings[i][0].shape[1]
        #    depth = encodings[i][0].shape[2]
        #    paddingHeight = maxHeight - height
        #    paddingWidth = maxWidth - width
        #    paddingDepth = maxDepth - depth
        #    if(center):
        #        priorH = paddingHeight // 2
        #        postH = paddingHeight - priorH
        #        priorW = paddingWidth // 2
        #        postW = paddingWidth - priorW
        #        priorD = paddingDepth // 2
        #        postD = paddingDepth - priorD
        #        encodings[i][0] = np.pad(encodings[i][0],((priorH,postH),(priorW,postW),(priorD,postD)),"constant",constant_values=pad_val)
        #    else:
        #        print("Error: currently center must be set to True")
        #        sys.exit(1)
        #    edge_encodings.append(encodings[i][0])

        ## normalize mutation encodings
        #mutation_encodings = []
        #maxHeight = int(maxTsTableSize[2])
        #maxWidth = int(maxTsTableSize[0])
        #for i in range(len(encodings)):
        #    height = encodings[i][1].shape[0]
        #    width = encodings[i][1].shape[1]
        #    paddingHeight = maxHeight - height
        #    paddingWidth = maxWidth - width
        #    if(center):
        #        priorH = paddingHeight // 2
        #        postH = paddingHeight - priorH
        #        priorW = paddingWidth // 2
        #        postW = paddingWidth - priorW
        #        encodings[i][1] = np.pad(encodings[i][1],((priorH,postH),(priorW,postW),(0,0)),"constant",constant_values=pad_val)
        #    else:
        #        print("Error: currently center must be set to True")
        #        sys.exit(1)
        #    mutation_encodings.append(encodings[i][1])

        #edge_encodings = np.array(edge_encodings)
        #mutation_encodings = np.array(mutation_encodings)

        #if(frameWidth):
        #    fw = frameWidth
        #    edge_encodings = np.pad(edge_encodings,((0,0),(fw,fw),(fw,fw),(fw,fw)),"constant",constant_values=pad_val)
        #    mutation_encodings = np.pad(mutation_encodings,((0,0),(fw,fw),(fw,fw),(fw,fw)),"constant",constant_values=pad_val)

        #return [edge_encodings, mutation_encodings]



        #### mutation matrix and edgeXconstant matrix
        ## normalize edge encodings
        #edge_encodings = []
        #maxHeight = int(maxTsTableSize[1])
        #pad_val = 0.0
        #for i in range(len(encodings)):
        #    height = encodings[i][0].shape[0]
        #    paddingHeight = maxHeight - height
        #    if(center):
        #        priorH = paddingHeight // 2
        #        postH = paddingHeight - priorH
        #        encodings[i][0] = np.pad(encodings[i][0],((priorH,postH),(0,0)),"constant",constant_values=pad_val)
        #    else:
        #        print("Error: currently center must be set to True")
        #        sys.exit(1)
        #    edge_encodings.append(encodings[i][0])

        ## normalize mutation encodings
        #mutation_encodings = []
        #maxHeight = int(maxTsTableSize[2])
        #maxWidth = int(maxTsTableSize[0])
        #for i in range(len(encodings)):
        #    height = encodings[i][1].shape[0]
        #    width = encodings[i][1].shape[1]
        #    paddingHeight = maxHeight - height
        #    paddingWidth = maxWidth - width
        #    if(center):
        #        priorH = paddingHeight // 2
        #        postH = paddingHeight - priorH
        #        priorW = paddingWidth // 2
        #        postW = paddingWidth - priorW
        #        encodings[i][1] = np.pad(encodings[i][1],((priorH,postH),(priorW,postW),(0,0)),"constant",constant_values=pad_val)
        #    else:
        #        print("Error: currently center must be set to True")
        #        sys.exit(1)
        #    mutation_encodings.append(encodings[i][1])

        #edge_encodings = np.array(edge_encodings)
        #mutation_encodings = np.array(mutation_encodings)

        #if(frameWidth):
        #    fw = frameWidth
        #    edge_encodings = np.pad(edge_encodings,((0,0),(fw,fw),(fw,fw),(fw,fw)),"constant",constant_values=pad_val)
        #    mutation_encodings = np.pad(mutation_encodings,((0,0),(fw,fw),(fw,fw),(fw,fw)),"constant",constant_values=pad_val)

        #return [edge_encodings, mutation_encodings]



        #### mutation matrix, marginal_trees matrix, tree spans, site spans
        ## normalize edge encodings
        #edge_encodings = []
        #mutation_encodings = []
        #tree_spans = []
        #site_spans = []
        #maxNodes = int(maxTsTableSize[0])
        #maxEdges = int(maxTsTableSize[1])
        #maxSites = int(maxTsTableSize[2])
        #maxMutations = int(maxTsTableSize[3])
        #maxTrees = int(maxTsTableSize[4])
        #pad_val = 0.0
        #for i in range(len(encodings)):
        #    # edge_encoding
        #    height = encodings[i][0].shape[0]
        #    width = encodings[i][0].shape[1]
        #    depth = encodings[i][0].shape[2]
        #    paddingHeight = maxTrees - height
        #    paddingWidth = maxNodes - width
        #    paddingDepth = maxNodes - depth
        #    if(center):
        #        priorH = paddingHeight // 2
        #        postH = paddingHeight - priorH
        #        priorW = paddingWidth // 2
        #        postW = paddingWidth - priorW
        #        priorD = paddingDepth // 2
        #        postD = paddingDepth - priorD
        #    else:
        #        print("Error: currently center must be set to True")
        #        sys.exit(1)
        #    encodings[i][0] = np.pad(encodings[i][0],((priorH,postH),(priorW,postW),(priorD,postD)),"constant",constant_values=pad_val)
        #    edge_encodings.append(encodings[i][0])

        #    # mutation_encoding
        #    height = encodings[i][1].shape[0]
        #    width = encodings[i][1].shape[1]
        #    paddingHeight = maxSites - height
        #    paddingWidth = maxNodes - width - 1
        #    if(center):
        #        priorH = paddingHeight // 2
        #        postH = paddingHeight - priorH
        #        priorW = paddingWidth // 2
        #        postW = paddingWidth - priorW
        #    encodings[i][1] = np.pad(encodings[i][1],((priorH,postH),(priorW,postW),(0,0)),"constant",constant_values=pad_val)
        #    mutation_encodings.append(encodings[i][1])

        #    # tree spans
        #    height = encodings[i][2].shape[0]
        #    paddingHeight = maxTrees - height
        #    if(center):
        #        priorH = paddingHeight // 2
        #        postH = paddingHeight - priorH
        #    encodings[i][2] = np.pad(encodings[i][2],((priorH,postH)),"constant",constant_values=pad_val)
        #    tree_spans.append(encodings[i][2])

        #    # site spans
        #    height = encodings[i][3].shape[0]
        #    paddingHeight = maxSites - height
        #    if(center):
        #        priorH = paddingHeight // 2
        #        postH = paddingHeight - priorH
        #    encodings[i][3] = np.pad(encodings[i][3],((priorH,postH)),"constant",constant_values=pad_val)
        #    site_spans.append([encodings[i][3]])

        ## convert to numpy arrays
        #edge_encodings = np.array(edge_encodings)
        #mutation_encodings = np.array(mutation_encodings)
        #tree_spans = np.array(tree_spans)
        #site_spans = np.array(site_spans)

        #if(frameWidth):
        #    fw = frameWidth
        #    edge_encodings = np.pad(edge_encodings,((0,0),(fw,fw),(fw,fw),(fw,fw)),"constant",constant_values=pad_val)
        #    mutation_encodings = np.pad(mutation_encodings,((0,0),(fw,fw),(fw,fw),(fw,fw)),"constant",constant_values=pad_val)
        #    tree_spans = np.pad(tree_spans,((0,0),(fw,fw)),"constant",constant_values=pad_val)
        #    site_spans = np.pad(site_spans,((0,0),(fw,fw)),"constant",constant_values=pad_val)

        #return [edge_encodings, mutation_encodings, tree_spans, site_spans]



        ### mutation matrix, marginal_trees matrix, tree spans, site spans
        # normalize edge encodings
        edge_encodings = []
        mutation_encodings = []
        tree_spans = []
        site_spans = []
        maxNodes = int(maxTsTableSize[0])
        maxEdges = int(maxTsTableSize[1])
        maxSites = int(maxTsTableSize[2])
        maxMutations = int(maxTsTableSize[3])
        maxTrees = int(maxTsTableSize[4])
        pad_val = 0.0
        for i in range(len(encodings)):
            # edge_encoding
            height = encodings[i][0].shape[0]
            width = encodings[i][0].shape[1]
            paddingHeight = maxTrees - height
            paddingWidth = maxEdges - width
            if(center):
                priorH = paddingHeight // 2
                postH = paddingHeight - priorH
                priorW = paddingWidth // 2
                postW = paddingWidth - priorW
            else:
                print("Error: currently center must be set to True")
                sys.exit(1)
            encodings[i][0] = np.pad(encodings[i][0],((priorH,postH),(priorW,postW)),"constant",constant_values=pad_val)
            edge_encodings.append(encodings[i][0])

            # mutation_encoding
            height = encodings[i][1].shape[0]
            width = encodings[i][1].shape[1]
            paddingHeight = maxSites - height
            paddingWidth = maxNodes - width - 1
            if(center):
                priorH = paddingHeight // 2
                postH = paddingHeight - priorH
                priorW = paddingWidth // 2
                postW = paddingWidth - priorW
            encodings[i][1] = np.pad(encodings[i][1],((priorH,postH),(priorW,postW)),"constant",constant_values=pad_val)
            mutation_encodings.append(encodings[i][1])

            # tree spans
            height = encodings[i][2].shape[0]
            paddingHeight = maxTrees - height
            if(center):
                priorH = paddingHeight // 2
                postH = paddingHeight - priorH
            encodings[i][2] = np.pad(encodings[i][2],((priorH,postH)),"constant",constant_values=pad_val)
            tree_spans.append(encodings[i][2])

            # site spans
            height = encodings[i][3].shape[0]
            paddingHeight = maxSites - height
            if(center):
                priorH = paddingHeight // 2
                postH = paddingHeight - priorH
            encodings[i][3] = np.pad(encodings[i][3],((priorH,postH)),"constant",constant_values=pad_val)
            site_spans.append(encodings[i][3])

        # convert to numpy arrays
        edge_encodings = np.array(edge_encodings)
        mutation_encodings = np.array(mutation_encodings)
        tree_spans = np.array(tree_spans)
        site_spans = np.array(site_spans)

        if(frameWidth):
            fw = frameWidth
            edge_encodings = np.pad(edge_encodings,((0,0),(fw,fw),(fw,fw)),"constant",constant_values=pad_val)
            mutation_encodings = np.pad(mutation_encodings,((0,0),(fw,fw),(fw,fw)),"constant",constant_values=pad_val)
            tree_spans = np.pad(tree_spans,((0,0),(fw,fw)),"constant",constant_values=pad_val)
            site_spans = np.pad(site_spans,((0,0),(fw,fw)),"constant",constant_values=pad_val)

        return [edge_encodings, mutation_encodings,
                tree_spans.reshape((tree_spans.shape[0],tree_spans.shape[1],1)),
                site_spans.reshape((site_spans.shape[0],site_spans.shape[1],1))]


    def pad_HapsPos(self,haplotypes,positions,maxSNPs=None,frameWidth=0,center=False):
        '''
        pads the haplotype and positions tensors
        to be uniform with the largest tensor
        '''

        haps = haplotypes
        pos = positions

        #Normalize the shape of all haplotype vectors with padding
        for i in range(len(haps)):
            numSNPs = haps[i].shape[0]
            paddingLen = maxSNPs - numSNPs
            if(center):
                prior = paddingLen // 2
                post = paddingLen - prior
                haps[i] = np.pad(haps[i],((prior,post),(0,0)),"constant",constant_values=2.0)
                pos[i] = np.pad(pos[i],(prior,post),"constant",constant_values=-1.0)

            else:
                if(paddingLen < 0):
                    haps[i] = np.pad(haps[i],((0,0),(0,0)),"constant",constant_values=2.0)[:paddingLen]
                    pos[i] = np.pad(pos[i],(0,0),"constant",constant_values=-1.0)[:paddingLen]
                else:
                    haps[i] = np.pad(haps[i],((0,paddingLen),(0,0)),"constant",constant_values=2.0)
                    pos[i] = np.pad(pos[i],(0,paddingLen),"constant",constant_values=-1.0)

        haps = np.array(haps,dtype='float32')
        pos = np.array(pos,dtype='float32')

        if(frameWidth):
            fw = frameWidth
            haps = np.pad(haps,((0,0),(fw,fw),(fw,fw)),"constant",constant_values=2.0)
            pos = np.pad(pos,((0,0),(fw,fw)),"constant",constant_values=-1.0)

        return haps,pos


    def shuffleIndividuals(self,x):
        t = np.arange(x.shape[1])
        np.random.shuffle(t)
        return x[:,t]


    def normalizeTargets(self):
        """
        normalize and stack all targets in the rawTargets list
        """
        norm = self.targetNormalization

        nTargets = []
        if(norm == 'zscore'):
            for i in range(len(self.rawTargets)):
                targets = copy.deepcopy(self.rawTargets[i])
                tar_mean = np.mean(targets, axis=0)
                tar_sd = np.std(targets, axis=0)
                targets -= tar_mean
                targets = np.divide(targets, tar_sd, out=np.zeros_like(targets), where=tar_sd != 0)
                nTargets.append(targets)
        elif(norm == 'binary'):
            for i in range(len(self.rawTargets)):
                targets = copy.deepcopy(self.rawTargets[i])
                nTargets.append(np.where(targets > 0, 1, targets))
        nTargets = np.stack(nTargets, axis=-1)
        return nTargets


    def on_epoch_end(self):

        if(self.shuffleExamples):
            np.random.shuffle(self.indices)


    def __len__(self):
        return int(np.floor(self.numReps/self.batch_size))


    def __getitem__(self, idx):

        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        if self.ReLERNN:
            x, y = self.__data_generation_ReLERNN(indices)
        else:
            x, y = self.__data_generation(indices)
        return x,y


    def __data_generation(self, batchTreeIndices):

        targets = np.array([t for t in self.normalizedTargets[batchTreeIndices]])
        encodings = []
        times = []
        for treeIndex in batchTreeIndices:
            tsFilePath = os.path.join(self.treesDirectory,"{}.trees".format(treeIndex))
            ts = tskit.load(tsFilePath)
            #cptdFilePath = os.path.join(self.treesDirectory,"{}.child_parent_treeID_dict".format(treeIndex))
            #cptd = pickle.load(open(cptdFilePath,"rb"))
            tsTensor = TsEncoder(ts, width=1000)
            tsTensor.add_node_time_layer()
            #tsTensor.add_parent_pointer()
            #tsTensor.add_branch_length_layer()
            encoding = tsTensor.get_encoding()
            encodings.append(encoding)
        encodings = np.array(encodings, dtype='float64')
        #encodings = self.pad_encodings(encodings, maxTsTableSize = self.maxTsTableSize, frameWidth=self.frameWidth, center=self.center)

        return encodings, targets


    def __data_generation_ReLERNN(self, batchTreeIndices):

        targets = np.array([t for t in self.normalizedTargets[batchTreeIndices]])
        haps = []
        pos = []
        for treeindex in batchTreeIndices:
            hfilepath = os.path.join(self.treesDirectory,str(treeindex) + "_haps.npy")
            pfilepath = os.path.join(self.treesDirectory,str(treeindex) + "_pos.npy")
            h = np.load(hfilepath)
            p = np.load(pfilepath)
            haps.append(h)
            pos.append(p)

        #if(self.reallinepos): #True was always used
        for p in range(len(pos)):
            pos[p] = pos[p] / self.infoDir["chromLength"]

        #if(self.shuffleinds): # True was always used
        for i in range(len(haps)):
            haps[i] = self.shuffleIndividuals(haps[i])

        # pad
        haps,pos = self.pad_HapsPos(haps,pos,
            maxSNPs=int(self.maxTsTableSize[2]),
            frameWidth=5,
            center=self.center)

        pos=np.where(pos == -1.0, 0,pos)
        haps=np.where(haps < 1.0, -1, haps)
        haps=np.where(haps > 1.0, 0, haps)
        haps=np.where(haps == 1.0, 1, haps)

        return [haps,pos], targets
