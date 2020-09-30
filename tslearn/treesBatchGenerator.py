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
            targetNormalization = 'zscore',
            batchSize=64,
            frameWidth=0,
            center=False,
            shuffleExamples = False,
            maxTsTableSize = None,
            numReps = None,
            rawTargets = None
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


        if(targetNormalization != None):
            self.normalizedTargets = self.normalizeTargets()

        if(shuffleExamples):
            np.random.shuffle(self.indices)


    def pad_encodings(self, encodings, maxTsTableSize=None, frameWidth=0, center=False):
        """
        pads all encodings to the max size of the tree sequence tables
        maxTsTableSize corresponds to the max lengths of of the
        [nodes, edges, sites, and mutations] table
        """

        maxHeight = int(maxTsTableSize[2])
        maxWidth = int(maxTsTableSize[0])
        pad_val = 0.0
        for i in range(len(encodings)):
            height = encodings[i].shape[0]
            width = encodings[i].shape[1]
            paddingHeight = maxHeight - height
            paddingWidth = maxWidth - width
            if(center):
                priorH = paddingHeight // 2
                postH = paddingHeight - priorH
                priorW = paddingWidth // 2
                postW = paddingWidth - priorW
                encodings[i] = np.pad(encodings[i],((priorH,postH),(priorW,postW),(0,0)),"constant",constant_values=pad_val)
            else:
                print("Error: currently center must be set to True")
                sys.exit(1)
        encodings = np.array(encodings)

        if(frameWidth):
            fw = frameWidth
            encodings = np.pad(encodings,((0,0),(fw,fw),(fw,fw),(fw,fw)),"constant",constant_values=pad_val)

        return encodings


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
        nTargets = np.stack(nTargets, axis=-1)

        return nTargets


    def on_epoch_end(self):

        if(self.shuffleExamples):
            np.random.shuffle(self.indices)


    def __len__(self):
        return int(np.floor(self.numReps/self.batch_size))


    def __getitem__(self, idx):

        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        x, y = self.__data_generation(indices)
        return x,y


    def __data_generation(self, batchTreeIndices):

        targets = np.array([t for t in self.normalizedTargets[batchTreeIndices]])

        encodings = []
        for treeIndex in batchTreeIndices:
            tsFilePath = os.path.join(self.treesDirectory,"{}.trees".format(treeIndex))
            ts = tskit.load(tsFilePath)
            tsTensor = Encoder(ts)
            tsTensor.add_mutation_matrix()
            encoding = tsTensor.get_encoding(dtype="float")
            encodings.append(encoding)

        encodings = self.pad_encodings(encodings, maxTsTableSize = self.maxTsTableSize, frameWidth=self.frameWidth, center=self.center)

        return encodings, targets
