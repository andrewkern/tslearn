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
        seed = 12345,
        N = 2,
	Ne = 1e2,
        priorLowsRho = 0.0,
        priorLowsMu = 0.0,
        priorHighsRho = 1e-7,
        priorHighsMu = 1e-8,
        ChromosomeLength = 1e5,
        MspDemographics = None,
        ):

        self.seed = seed
        self.N = N
        self.Ne = Ne
        self.priorLowsRho = priorLowsRho
        self.priorHighsRho = priorHighsRho
        self.priorLowsMu = priorLowsMu
        self.priorHighsMu = priorHighsMu
        self.ChromosomeLength = ChromosomeLength
        self.MspDemographics = MspDemographics
        self.rho = None
        self.hotWin = None
        self.mu = None


    def runOneMsprimeSim(self,simNum,direc):
        '''
        run one msprime simulation and put the corresponding treeSequence in treesOutputFilePath

        (str,float,float)->None
        '''

        MR = self.mu[simNum]
        RR = self.rho[simNum]
        SEED = self.seed[simNum]

        if self.MspDemographics:
            DE = self.MspDemographics["demographic_events"]
            PC = self.MspDemographics["population_configurations"]
            MM = self.MspDemographics["migration_matrix"]
            ts = msprime.simulate(
                random_seed = SEED,
                length=self.ChromosomeLength,
                mutation_rate=MR,
                recombination_rate=RR,
                population_configurations = PC,
                migration_matrix = MM,
                demographic_events = DE
            )
        else:
            ts = msprime.simulate(
                random_seed = SEED,
                sample_size = self.N,
                Ne = self.Ne,
                length=self.ChromosomeLength,
                mutation_rate=MR,
                recombination_rate=RR
            )

        # dump trees
        tsFileName = os.path.join(direc, "{}.trees".format(simNum))
        ts.dump(tsFileName)

        # Return number of sites

        return [ts.num_nodes, ts.num_edges, ts.num_sites, ts.num_mutations]


    def simulateAndProduceTrees(self,direc,numReps,simulator,nProc=1):
        '''
        determine which simulator to use then populate

        (str,str) -> None
        '''
        self.numReps = numReps
        self.seed=np.repeat(self.seed,numReps)
        self.rho=np.empty(numReps)
        for i in range(numReps):
            randomTargetParameter = np.random.uniform(self.priorLowsRho,self.priorHighsRho)
            self.rho[i] = randomTargetParameter

        self.mu=np.empty(numReps)
        for i in range(numReps):
            randomTargetParameter = np.random.uniform(self.priorLowsMu,self.priorHighsMu)
            self.mu[i] = randomTargetParameter

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
        for i in range(result_q.qsize()):
            item = result_q.get()
            self.numNodes[item[0]]=item[1][0]
            self.numEdges[item[0]]=item[1][1]
            self.numSites[item[0]]=item[1][2]
            self.numMutations[item[0]]=item[1][3]

        infofile = open(os.path.join(direc,"info.p"),"wb")
        pickle.dump(self.__dict__,infofile)
        infofile.close()

        for p in pids:
            p.terminate()

        return np.array([self.numNodes.max(),
            self.numEdges.max(),
            self.numSites.max(),
            self.numMutations.max()]
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
