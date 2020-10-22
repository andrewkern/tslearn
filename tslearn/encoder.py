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
        self.mutationEncoding = None
        self.edgeEncoding = None


    #def add_mutation_matrix(self):
    #    """
    #    return the matrix of shape(num_sites,num_nodes,max_num_alleles)
    #    currently assume max_num_alleles = 2 (i.e. 0 and 1)
    #    """

    #    self.mutationEncoding = np.zeros((self.ts.num_sites, self.ts.num_nodes-1, 2), dtype=self.datatype)
    #    self.mutationSpanEncoding = np.zeros(self.ts.num_sites, dtype=self.datatype)

    #    init_pos = 0
    #    for i, site in enumerate(self.ts.sites()):
    #        self.mutationSpanEncoding[int(site.id)] = site.position - init_pos
    #        init_pos = site.position

    #    for i, mutation in enumerate(self.ts.mutations()):
    #        self.mutationEncoding[int(mutation.site), int(mutation.node), int(mutation.derived_state)] += 1

    #    return None


    def add_mutation_matrix(self):
        """
        OPTIMIZED FOR LSTM
        returns a matrix of shape(num_sites,num_nodes)
        where cell values are the number of total mutations above each node
        (includes derived and back mutations)
        also returns a matrix of shape(num_sites)
        where values are the distances between sites
        """

        self.mutationEncoding = np.zeros((self.ts.num_sites, self.ts.num_nodes-1), dtype=self.datatype)
        self.mutationSpanEncoding = np.zeros(self.ts.num_sites, dtype=self.datatype)

        init_pos = 0
        for i, site in enumerate(self.ts.sites()):
            self.mutationSpanEncoding[int(site.id)] = site.position - init_pos
            init_pos = site.position

        for i, mutation in enumerate(self.ts.mutations()):
            self.mutationEncoding[int(mutation.site), int(mutation.node)] += 1

        return None


    #def add_edge_matrix(self):
    #    """
    #    return the matrix of shape(num_edges,num_nodes+2,num_nodes)
    #    where each edge has a left and right position and a nodeXnode matrix
    #    where the length of each edge is entered the cells corresponding to
    #    the parent and child node
    #    """

    #    self.edgeEncoding = np.zeros((self.ts.num_edges, self.ts.num_nodes+2, self.ts.num_nodes), dtype=self.datatype)
    #    for i, edge in enumerate(self.ts.edges()):
    #        self.edgeEncoding[int(edge.id), 0, 0] = edge.left
    #        self.edgeEncoding[int(edge.id), 1, 0] = edge.right
    #        branch_length = self.ts.node(edge.parent).time - self.ts.node(edge.child).time
    #        self.edgeEncoding[int(edge.id), edge.parent+2, edge.child] = branch_length
    #        self.edgeEncoding[int(edge.id), edge.child+2, edge.parent] = branch_length

    #    return None


    #def add_edge_matrix(self):
    #    """
    #    return the matrix of shape(num_edges,5)
    #    where each edge has a left and right position, a length
    #    and the index of the parent and child node
    #    """

    #    self.edgeEncoding = np.zeros((self.ts.num_edges, 5), dtype=self.datatype)
    #    for i, edge in enumerate(self.ts.edges()):
    #        self.edgeEncoding[int(edge.id), 0] = edge.left
    #        self.edgeEncoding[int(edge.id), 1] = edge.right
    #        branch_length = self.ts.node(edge.parent).time - self.ts.node(edge.child).time
    #        self.edgeEncoding[int(edge.id), 2] = branch_length
    #        self.edgeEncoding[int(edge.id), 3] = edge.parent
    #        self.edgeEncoding[int(edge.id), 4] = edge.child

    #    return None


    #def add_edge_matrix(self):
    #    """
    #    return the matrix of shape(num_nodes,num_nodes,)
    #    where each edge has a left and right position, a length
    #    and the index of the parent and child node
    #    """

    #    self.edgeEncoding = np.zeros((self.ts.num_nodes, self.ts.num_nodes), dtype=self.datatype)
    #    marg_trees = {} #each marginal tree will have [index,edge_count]
    #    tree_index = 0
    #    for i, edge in enumerate(self.ts.edges()):
    #        print(edge)
    #        branch_length = self.ts.node(edge.parent).time - self.ts.node(edge.child).time
    #        try:
    #            marg_trees["{}:{}".format(edge.left, edge.right)][1] += 1
    #            idx = marg_trees["{}:{}".format(edge.left, edge.right)][0]
    #            if tree_index == 1:
    #                self.edgeEncoding[edge.parent, edge.child] = branch_length
    #                self.edgeEncoding[edge.child, edge.parent] = branch_length
    #            else:
    #                self.edgeEncoding[edge.parent, edge.child, idx] = branch_length
    #                self.edgeEncoding[edge.child, edge.parent, idx] = branch_length
    #        except KeyError:
    #            if tree_index == 0:
    #                marg_trees["{}:{}".format(edge.left, edge.right)] = [0, 1]
    #                self.edgeEncoding = np.zeros((self.ts.num_nodes, self.ts.num_nodes), dtype=self.datatype)
    #                self.edgeEncoding[edge.parent, edge.child] = branch_length
    #                self.edgeEncoding[edge.child, edge.parent] = branch_length
    #                tree_index += 1
    #            else:
    #                marg_trees["{}:{}".format(edge.left, edge.right)] = [tree_index, 1]
    #                layer = np.zeros((self.ts.num_nodes, self.ts.num_nodes), dtype=self.datatype)
    #                self.edgeEncoding = np.dstack((self.edgeEncoding, layer))
    #                self.edgeEncoding[edge.parent, edge.child, tree_index] = branch_length
    #                self.edgeEncoding[edge.child, edge.parent, tree_index] = branch_length
    #                tree_index += 1

    #    print("+++++++++++++++++++++++++++++++++++++")
    #    print(self.edgeEncoding.shape)
    #    print("self.ts.num_trees:",self.ts.num_trees)
    #    for tree in self.ts.trees():
    #        print(tree)
    #    print("\n\n")
    #    for tree in marg_trees:
    #        ar = tree.split(":")
    #        print("left: {},  right:{}".format(ar[0],ar[1]))
    #    sys.exit()
    #    return None



    #def add_edge_matrix(self, cptd):
    #    """
    #    return the matrix of shape(num_trees,num_nodes,num_nodes)
    #    where each tree has the branch lengths connecting the nodes
    #    """

    #    # cptd = child_parent_tree_dict, where each entry is "childID:parentID" = [list_of_treesID]
    #    self.edgeSpanEncoding = np.zeros(self.ts.num_trees, dtype=self.datatype)
    #    self.edgeEncoding = np.zeros((self.ts.num_trees, self.ts.num_nodes, self.ts.num_nodes), dtype=self.datatype)
    #    #child_parent_trees = {} #each entry is "child:parent" = [list_of_trees_by_index]
    #    #child_parent_trees = cptd #each entry is "child:parent" = [list_of_trees_by_index]
    #    #t1 = time.perf_counter()
    #    #for i, tree in enumerate(self.ts.trees()):
    #    #    for child in tree.parent_dict:
    #    #        pair_key = "{}:{}".format(child,tree.parent_dict[child])
    #    #        try:
    #    #            child_parent_trees[pair_key].append(i)
    #    #        except KeyError:
    #    #            child_parent_trees[pair_key] = [i]
    #    #t2 = time.perf_counter()
    #    #print("tree loop:",t2-t1)
    #    #t1 = time.perf_counter()
    #    for i, edge in enumerate(self.ts.edges()):
    #        branch_length = self.ts.node(edge.parent).time - self.ts.node(edge.child).time
    #        pair_key = "{}:{}".format(edge.child,edge.parent)
    #        for j, tree in enumerate(cptd[pair_key]):
    #            #print(cptd[pair_key])
    #            #sys.exit()
    #            self.edgeEncoding[tree[0], edge.child, edge.parent] = branch_length
    #            self.edgeEncoding[tree[0], edge.parent, edge.child] = branch_length
    #            self.edgeSpanEncoding[tree[0]] = tree[1]
    #    #t2 = time.perf_counter()
    #    #print("edge loop:",t2-t1)
    #    #print("+++++")
    #    #np.set_printoptions(threshold=np.inf)
    #    #print(self.edgeEncoding[0])
    #    #print(self.edgeSpanEncoding)
    #    #print(self.edgeSpanEncoding.shape)
    #    #sys.exit()

    #    return None


    def add_edge_matrix(self, cptd):
        """
        OPTIMIZED FOR LSTM
        returns a matrix of shape(num_trees,num_child_parent_pairs)
        where each tree has the branch lengths connecting the nodes
        also returns a matrix of shape(num_trees)
        where values are the spans of each tree
        """

        self.edgeSpanEncoding = np.zeros(self.ts.num_trees, dtype=self.datatype)
        self.edgeEncoding = np.zeros((self.ts.num_trees, self.ts.num_edges), dtype=self.datatype)
        for i, edge in enumerate(self.ts.edges()):
            branch_length = self.ts.node(edge.parent).time - self.ts.node(edge.child).time
            pair_key = "{}:{}".format(edge.child,edge.parent)
            for j, tree in enumerate(cptd[pair_key]):
                self.edgeEncoding[tree[0], edge.id] = branch_length
                self.edgeSpanEncoding[tree[0]] = tree[1]

        return None


    def get_encoding(self, dtype=None):
        """
        return the encodings of the TreeSequence
        """

        if dtype is not None:
            return [self.edgeEncoding.astype(dtype),self.mutationEncoding.astype(dtype),
                    self.edgeSpanEncoding.astype(dtype),self.mutationSpanEncoding.astype(dtype)]
        else:
            return [self.edgeEncoding,self.mutationEncoding,
                    self.edgeSpanEncoding, self.mutationSpanEncoding]
