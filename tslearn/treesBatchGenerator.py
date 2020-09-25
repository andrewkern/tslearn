'''
Authors: Jared Galloway, Jeff Adrion
'''

from tslearn.imports import *
from tslearn.tsEncoder import *

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
            maxLen=None,
            frameWidth=0,
            center=False,
            shuffleExamples = False
            ):

        self.treesDirectory = treesDirectory
        self.targetNormalization = targetNormalization
        self.batch_size = batchSize
        self.maxLen = maxLen
        self.frameWidth = frameWidth
        self.center = center
        infoFilename = os.path.join(self.treesDirectory,"info.p")
        self.infoDir = pickle.load(open(infoFilename,"rb"))
        self.indices = np.arange(self.infoDir["numReps"])
        self.shuffleExamples = shuffleExamples


        if(targetNormalization != None):
            self.normalizedTargets = self.normalizeTargets()

        if(shuffleExamples):
            np.random.shuffle(self.indices)


    def pad_encodings(self, trees, maxHeight=None, frameWidth=0, center=False):

        pad_val = 0.0
        for i in range(len(trees)):
            height = trees[i].shape[0]
            paddingLen = maxHeight - height
            if(center):
                prior = paddingLen // 2
                post = paddingLen - prior
                trees[i] = np.pad(trees[i],((prior,post),(0,0),(0,0)),"constant",constant_values=pad_val)
            else:
                if(paddingLen < 0):
                    trees[i] = np.pad(trees[i],((0,0),(0,0),(0,0)),"constant",constant_values=pad_val)[:paddingLen]
                else:
                    trees[i] = np.pad(trees[i],((0,paddingLen),(0,0),(0,0)),"constant",constant_values=pad_val)

        trees = np.array(trees)

        if(frameWidth):
            fw = frameWidth
            trees = np.pad(trees,((0,0),(fw,fw),(fw,fw),(fw,fw)),"constant",constant_values=pad_val)

        return trees


    def normalizeTargets(self):

        norm = self.targetNormalization
        nTargets_mu = copy.deepcopy(self.infoDir['mu'])
        nTargets_rho = copy.deepcopy(self.infoDir['rho'])

        if(norm == 'zscore'):
            #mu
            tar_mean_mu = np.mean(nTargets_mu,axis=0)
            tar_sd_mu = np.std(nTargets_mu,axis=0)
            nTargets_mu -= tar_mean_mu
            nTargets_mu = np.divide(nTargets_mu,tar_sd_mu,out=np.zeros_like(nTargets_mu),where=tar_sd_mu!=0)
            #rho
            tar_mean_rho = np.mean(nTargets_rho,axis=0)
            tar_sd_rho = np.std(nTargets_rho,axis=0)
            nTargets_rho -= tar_mean_rho
            nTargets_rho = np.divide(nTargets_rho,tar_sd_rho,out=np.zeros_like(nTargets_rho),where=tar_sd_rho!=0)

        ## stack targets
        nTargets = np.stack((nTargets_mu, nTargets_rho), axis=-1)

        return nTargets


    def on_epoch_end(self):

        if(self.shuffleExamples):
            np.random.shuffle(self.indices)


    def __len__(self):

        return int(np.floor(self.infoDir["numReps"]/self.batch_size))


    def __getitem__(self, idx):

        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        x, y = self.__data_generation(indices)
        return x,y


    def __data_generation(self, batchTreeIndices):

        encodings = []
        max_height = 0

        for treeIndex in batchTreeIndices:
            tsFilePath = os.path.join(self.treesDirectory,"{}.trees".format(treeIndex))
            ts = tskit.load(tsFilePath)
            encoder = TsEncoder(ts)
            encoder.add_one_to_one()
            encoder.normalize_layers(layers=[0])
            encoding = encoder.get_encoding(dtype='float')
            max_height = max(max_height, encoding.shape[0])
            encodings.append(encoding)

        respectiveNormalizedTargets = [t for t in self.normalizedTargets[batchTreeIndices]]
        targets = np.array(respectiveNormalizedTargets)
        encodings = self.pad_encodings(encodings, maxHeight = max_height, frameWidth=self.frameWidth, center=self.center)

        return encodings, targets
