'''
Authors: Jeff Adrion
'''

from tslearn.imports import *

class Encoder():
    """
    The class will include methods for generating
    the tensors used to train on tree sequences
    """

    def __init__(self,
                 treeSequence,
                 dtype=None):

        self.ts = treeSequence
        self.datatype = dtype
        if dtype is None:
            self.datatype = np.float64
        self.encoding = None
        self.layerIndex = -1

    def initialize_layer(self):
        """
        initialize layer
        """

        layer = np.zeros((self.ts.num_sites, self.ts.num_nodes, 2), dtype=self.datatype)

        if(self.layerIndex < 0):
            self.encoding = layer
            self.layerIndex = 0
        else:
            self.encoding = np.append(self.encoding, layer, axis=2)
            self.layerIndex += 1

        return None

    def add_mutation_matrix(self):
        """
        return the matrix of shape(num_sites,num_nodes,max_num_alleles)
        currently assume max_num_alleles = 2 (i.e. 0 and 1)
        """

        self.initialize_layer()

        for i, mutation in enumerate(self.ts.mutations()):
            self.encoding[int(mutation.site), int(mutation.node), int(mutation.derived_state)] += 1

        return None

    def get_encoding(self, dtype=None):
        """
        return the actual encoding of the TreeSequence
        """

        if dtype is not None:
            return self.encoding.astype(dtype)
        else:
            return self.encoding
