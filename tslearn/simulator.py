'''
Author: Jeff Adrion, Jared Galloway
'''

from tslearn.imports import *
from tslearn.helpers import *

class Simulator(object):
    '''

    The simulator class is a framework for running N simulations
    using Either msprime (coalescent) or SLiM (forward-moving)
    in parallel using python's multithreading package.

    With Specified parameters, the class Simulator() populates
    a directory with training, validation, and testing datasets.
    It stores the the treeSequences resulting from each simulation
    in a subdirectory respectfully labeled 'i.trees' where i is the
    i^th simulation.

    Included with each dataset this class produces an info.p
    in the subdirectory. This uses pickle to store a dictionary
    containing all the information for each simulation including the random
    target parameter which will be extracted for training.

    '''

    def __init__(self,
        seed = None,
        N = None,
	Ne = None,
        mrPrior = None,
        rrPrior = None,
        chromLength = None,
        epoch_N = None,
        epoch_t1 = None,
        epoch_t2 = None,
        ReLERNN = None
        ):

        self.seed = seed
        self.N = N
        self.Ne = Ne
        self.mrPrior = mrPrior
        self.rrPrior = rrPrior
        self.chromLength = chromLength
        self.epoch_N = epoch_N
        self.epoch_t1 = epoch_t1
        self.epoch_t2 = epoch_t2
        self.ReLERNN = ReLERNN

        assert min(self.epoch_t2) > max(self.epoch_t1)

    def runOneMsprimeSim(self,simNum,direc):
        '''
        run one msprime simulation and put the corresponding treeSequence in treesOutputFilePath

        (str,float,float)->None
        '''

        MR = self.mu[simNum]
        RR = self.rho[simNum]
        SEED = self.seed[simNum]

        if self.epoch_N:
            t1 = self.epoch_t1_coal[simNum] * 4 * self.Ne
            t2 = self.epoch_t2_coal[simNum] * 4 * self.Ne
            PC = [msprime.PopulationConfiguration(
                initial_size = int(self.epoch_N_modern[simNum]),
                sample_size=self.N)]
            DE = [msprime.PopulationParametersChange(
                time=t1, initial_size=int(self.epoch_N_intermediate[simNum]), population_id=0),
                msprime.PopulationParametersChange(
                    time=t2, initial_size=int(self.epoch_N_ancestral[simNum]), population_id=0)]
            #DD = msprime.DemographyDebugger(
            #        population_configurations=PC,
            #        demographic_events=DE)
            #DD.print_history()
            ts = msprime.simulate(
                random_seed = SEED,
                length=self.chromLength,
                mutation_rate=MR,
                recombination_rate=RR,
                population_configurations = PC,
                demographic_events = DE
            )
        else:
            ts = msprime.simulate(
                random_seed = SEED,
                sample_size = self.N,
                Ne = self.Ne,
                length=self.chromLength,
                mutation_rate=MR,
                recombination_rate=RR
            )

        # dump trees
        tsFileName = os.path.join(direc, "{}.trees".format(simNum))
        ts.dump(tsFileName)

        '''
        dump dict where child-parent pairs have a value of
        all treeIDs with the focal branch as a list, [[treeID, tree.span],])
        '''
        child_parent_trees = {} #each entry is "child:parent" = [list_of_trees_by_index]
        for i, tree in enumerate(ts.trees()):
            for child in tree.parent_dict:
                pair_key = "{}:{}".format(child,tree.parent_dict[child])
                try:
                    child_parent_trees[pair_key].append([i,tree.span])
                except KeyError:
                    child_parent_trees[pair_key] = [[i,tree.span]]
        cptdFileName = os.path.join(direc, "{}.child_parent_treeID_dict".format(simNum))
        pickle.dump(child_parent_trees, open(cptdFileName,"wb"))

        if self.ReLERNN:
            H = ts.genotype_matrix()
            P = np.array([s.position for s in ts.sites()],dtype='float32')
            Hname = str(simNum) + "_haps.npy"
            Hpath = os.path.join(direc,Hname)
            Pname = str(simNum) + "_pos.npy"
            Ppath = os.path.join(direc,Pname)
            np.save(Hpath,H)
            np.save(Ppath,P)

        # Return tree sequence table stats
        return [ts.num_nodes, ts.num_edges, ts.num_sites, ts.num_mutations, ts.num_trees]


    def simulateAndProduceTrees(self,direc,numReps,simulator,nProc=1):
        '''
        determine which simulator to use then populate

        (str,str) -> None
        '''
        self.numReps = numReps
        self.seed=np.repeat(self.seed,numReps)
        self.rho=np.empty(numReps)
        for i in range(numReps):
            randomTargetParameter = np.random.uniform(self.rrPrior[0],self.rrPrior[1])
            self.rho[i] = randomTargetParameter

        self.mu=np.empty(numReps)
        for i in range(numReps):
            randomTargetParameter = np.random.uniform(self.mrPrior[0],self.mrPrior[1])
            self.mu[i] = randomTargetParameter

        # if n-epoch model:
        if self.epoch_N:
            self.epoch_N_modern = np.empty(numReps)
            self.epoch_N_intermediate = np.empty(numReps)
            self.epoch_N_ancestral = np.empty(numReps)
            self.epoch_t1_coal = np.empty(numReps)
            self.epoch_t2_coal = np.empty(numReps)
            for i in range(numReps):
                randomTargetParameter = np.random.uniform(self.epoch_N[0], self.epoch_N[1])
                self.epoch_N_modern[i] = int(randomTargetParameter)
                randomTargetParameter = np.random.uniform(self.epoch_N[0], self.epoch_N[1])
                self.epoch_N_intermediate[i] = int(randomTargetParameter)
                randomTargetParameter = np.random.uniform(self.epoch_N[0], self.epoch_N[1])
                self.epoch_N_ancestral[i] = int(randomTargetParameter)
                randomTargetParameter = np.random.uniform(self.epoch_t1[0], self.epoch_t1[1])
                self.epoch_t1_coal[i] = randomTargetParameter
                randomTargetParameter = np.random.uniform(self.epoch_t2[0], self.epoch_t2[1])
                self.epoch_t2_coal[i] = randomTargetParameter

        try:
            assert((simulator=='msprime') | (simulator=='SLiM'))
        except:
            print("Sorry, only 'msprime' & 'SLiM' are supported simulators")
            exit()

        #Pretty straitforward, create the directory passed if it doesn't exits
        if not os.path.exists(direc):
            print("directory '",direc,"' does not exist, creating it")
            os.makedirs(direc)

        # partition data for multiprocessing
        mpID = range(numReps)
        task_q = mp.JoinableQueue()
        result_q = mp.Queue()
        params=[simulator, direc]

        # do the work
        print("Simulate...")
        pids = create_procs(nProc, task_q, result_q, params, self.worker_simulate)
        assign_task(mpID, task_q, nProc)
        try:
            task_q.join()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            sys.exit(0)

        self.numNodes=np.empty(numReps,dtype="int64")
        self.numEdges=np.empty(numReps,dtype="int64")
        self.numSites=np.empty(numReps,dtype="int64")
        self.numMutations=np.empty(numReps,dtype="int64")
        self.numTrees=np.empty(numReps,dtype="int64")
        for i in range(result_q.qsize()):
            item = result_q.get()
            self.numNodes[item[0]]=item[1][0]
            self.numEdges[item[0]]=item[1][1]
            self.numSites[item[0]]=item[1][2]
            self.numMutations[item[0]]=item[1][3]
            self.numTrees[item[0]]=item[1][4]

        infofile = open(os.path.join(direc,"info.p"),"wb")
        pickle.dump(self.__dict__,infofile)
        infofile.close()

        for p in pids:
            p.terminate()

        return np.array([self.numNodes.max(),
            self.numEdges.max(),
            self.numSites.max(),
            self.numMutations.max(),
            self.numTrees.max()]
            )


    def worker_simulate(self, task_q, result_q, params):
        while True:
            try:
                mpID, nth_job = task_q.get()
                #unpack parameters
                simulator, direc = params
                for i in mpID:
                        result_q.put([i,self.runOneMsprimeSim(i,direc)])
            finally:
                task_q.task_done()
