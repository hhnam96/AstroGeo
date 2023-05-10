import numpy as np

def selection_tournament(fits, n, k=2, elitism=True):
    """
    input: fits - the fitness of the populations
           k - size of the tournament
           n - size of the selected individuals
    Randomly select k participants for the tournaments, pick the best amongst k participants
    Repeat n times the tournaments to obtain n winners
    If elitism is True: always retain the best individual
    output: indexes of n selected individuals by the tournament
    """
    inds = np.arange(len(fits))
    inds_tnm = np.random.choice(inds, (n,k))
    winner_results = np.argmax(fits[inds_tnm], 1)
    inds_winner =np.choose(winner_results, inds_tnm.T)
    if elitism:
        arg_best = np.argmax(fits)
        if arg_best not in inds_winner:
            inds_winner[np.random.choice(n)] = arg_best
    return inds_winner


def crossover_blend(x1, x2, alpha=0.5, xrange=[0, 2]):
    """blend crossover between two array

    Args:
        x1 (n): chromosome/genotype 1 
        x2 (n): chromosome/geotype 2
        alpha (float, optional): _description_. Defaults to 0.5.

    Returns:
        two offsprings (y1(n), y2(n), each GENE (example: x1[0]) of each children is drawn randomly in the range
            [x1' - alpha*(x2'-x1'), x2' + alpha*(x2'-x1')]
            where x1',x2' = min(x1,x2), max(x1,x2)) 
    """
    assert len(x1) == len(x2)
    n = len(x1)
    x12 = np.vstack([x1,x2])
    min_x12 = np.min(x12, axis=0)
    max_x12 = np.max(x12, axis=0)
    range_x12 = max_x12-min_x12
    lower = np.max([min_x12-range_x12*alpha , np.ones(n)*xrange[0]],axis=0)
    upper = np.min([max_x12+range_x12*alpha , np.ones(n)*xrange[1]],axis=0)
    offsprings = [np.random.rand(n)*(upper - lower) + lower for _ in range(2)] 
    return offsprings

def crossover_pop(X, fits, crossover_operator=crossover_blend, *args, **kwargs):
    """_summary_
        Perform 2d 2-point crossover of 2 neighboring individuals in population 
        with probability pc = (fmax - f')/(fmax - <f>)        if f' >= <f>
                            = 1                             if f' <  <f>
        where   f' is the larger fitness of the two individuals
                fmax is the maximum fitness in the population
                <f> is the average fitness in the population
        Meaning:    All the individuals having fitness lower than average will be crossed, 
                    Those that good fitness are crossed less (to preserve good genes)
                    Population is crossed more when population getting stagnant (fmax is close to <f>)
                 
        Ref: Adaptive Probabilities of Crossover and Mutation (Srinivas and Patnaik 1994)
    Args:
        X (list[N_population][2, N_genes]): collection of individuals in the population
        metric_data (array[N_population]): metric mesures the fitness of individuals in the population
    Returns:
        _type_: _description_
    """
    fp = np.max(np.vstack([fits[::2],fits[1::2]]),axis=0)
    fmean = fits.mean(); fmax = fits.max()
    pc = np.ones(len(fp))
    eps = 1e-5
    pc[fp>=fmean] = (fmax-fp[fp>=fmean])/(fmax-fmean + eps) 
    for i in range(len(pc)):
        if np.random.rand() < pc[i]:
            X[i*2:i*2+2] = crossover_operator(X[i*2], X[i*2+1], *args, **kwargs)
    return X


def mutation_poly(x1, pm, xrange=[0,1], eta=50):
    """
    Polynomial Mutation (Deb and Deb 2014)
    input: x1(n, ) is a chromosome containg n genes
    pm is probability of mutation for each  genes
    output: new x1
    """
    x1_m = x1.copy()
    n = len(x1)
    bool_m = np.random.rand(n) < pm
    arg_m  = np.argwhere(bool_m)[:,0]
    u = np.random.rand(len(arg_m))
    arg_m_L = arg_m[u<=0.5]
    arg_m_R = arg_m[u>0.5]
    delta_L = (2*u[u<=0.5])**(1/(1+eta))-1
    delta_R = 1-(2*(1-u[u>0.5]))**(1/(1+eta))
    x1_m[arg_m_L] = x1_m[arg_m_L] + delta_L*(x1_m[arg_m_L] - xrange[0])
    x1_m[arg_m_R] = x1_m[arg_m_R] + delta_R*(-x1_m[arg_m_R] + xrange[1])
    return x1_m

def mutation_pop(X, fits, mutation_operator=mutation_poly, *arg, **kwargs):
    """_summary_
        Perform mutation_v2 for each individual in the population X
        with probability pm = (fmax - f')/(fmax - <f>)        if f' >= <f>
                            = 1                             if f' <  <f>
        where   f' is the larger fitness of the two individuals
                fmax is the maximum fitness in the population
                <f> is the average fitness in the population
        Meaning:    All the individuals having fitness lower than average will be crossed, 
                    Those that good fitness are crossed less (to preserve good genes)
                    Population is crossed more when population getting stagnant (fmax is close to <f>)
                 
        Ref: Adaptive Probabilities of Crossover and Mutation (Srinivas and Patnaik 1994)
    Args:
        X (list[N_population][2, N_genes]): collection of individuals in the population
        metric_data (array[N_population]): metric mesures the fitness of individuals in the population
    Returns:
        _type_: _description_
    """
    
    fmean = fits.mean(); fmax=fits.max()
    pm = np.ones(len(fits))*0.5
    eps=1e-5
    pm[fits>=fmean] = 0.5*(fmax - fits[fits>=fmean]) / (fmax - fmean + eps) 
#     print(pm)
    Xm = [mutation_operator(X[i], pm[i], *arg, **kwargs) for i in range(len(X))]
    return Xm
