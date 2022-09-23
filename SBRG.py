import random
from operator import attrgetter
from itertools import combinations, product
from copy import deepcopy
from numba import jit

from qutip import *
# helpful function
def lorentz_line(w, x, delta=0.01):
    return delta/(delta**2 + (w-x)**2)/np.pi
''' Mat: tensor product of Pauli matrices
Mat.Xs :: frozenset : collection of sites of X gates
Mat.Zs :: frozenset : collection of sites of Z gates
'''
class Mat:
    def __init__(self, Xs, Zs):
        self.Xs = Xs
        self.Zs = Zs
        self._ipower = None
        self._key = None
    def __repr__(self):
        return '<Xs:%s Zs:%s>' % (sorted(list(self.Xs)), sorted(list(self.Zs)))
    def __hash__(self):
        if self._key is None:
            self._key = hash((self.Xs, self.Zs))
        return self._key
    def __eq__(self, other):
        return self.Xs == other.Xs and self.Zs == other.Zs
    def __neq__(self, other):
        return self.Xs != other.Xs or self.Zs != other.Zs
    def ipower(self): # number of overlap between Xs and Zs (num of Y gates)
        if self._ipower is None: # if ipower has not been calculated
            self._ipower = len(self.Xs & self.Zs)
            # once calculated the result is stored in self._ipower
        return self._ipower
# use mkMat to construct Mat
def mkMat(*arg):
    l_arg = len(arg)
    if l_arg == 2:
        return Mat(frozenset(arg[0]),frozenset(arg[1]))
    elif l_arg == 1:
        inds = arg[0]
        Xs = set()
        Zs = set()
        if isinstance(inds, dict): # dict of inds rules
        # example: mkMat({i:mu, ...})
            for (i, mu) in inds.items():
                if mu == 1:
                    Xs.add(i)
                elif mu == 3:
                    Zs.add(i)
                elif mu == 2:
                    Xs.add(i)
                    Zs.add(i)
        elif isinstance(inds, (tuple, list)): # list of inds
        # example: mkMat([mu0, mu1, mu2, ...])
            for (i, mu) in enumerate(inds):
                if mu == 0:
                    continue
                elif mu == 1:
                    Xs.add(i)
                elif mu == 3:
                    Zs.add(i)
                elif mu == 2:
                    Xs.add(i)
                    Zs.add(i)
        return Mat(frozenset(Xs), frozenset(Zs))
    elif l_arg == 0: # empty Mat by mkMat()
        return Mat(frozenset(), frozenset())
    else:
        raise TypeError('mkMat expected at most 2 arguments, got %s.' % l_arg)
############################### HYH ###############################
class MajaronaRep:
    def __init__(self,gamma,gamma_t):
        self.gamma = gamma
        self.gamma_t = gamma_t
    def __repr__(self):
        return '<gamma:%s gamma\':%s>' % (sorted(list(self.gamma)), sorted(list(self.gamma_t)))
def mkMaj(Mat):
    tmp_gamma = set({})
    tmp_gamma_t = set({})
    for pauli in Mat.Xs:
        tmp_gamma = tmp_gamma^set({pauli})
        tmp_gamma_t = tmp_gamma_t^set({pauli})
    for pauli in Mat.Zs:
        tmp = {i for i in range(pauli)}
        tmp_gamma = tmp_gamma^tmp
        tmp_gamma = tmp_gamma^set({pauli})
        tmp_gamma_t = tmp_gamma_t^tmp
    return MajaronaRep(tmp_gamma,tmp_gamma_t)
############################### HYH ###############################


# commutativity check
def is_commute(mat1, mat2):
    return (len(mat1.Xs & mat2.Zs) - len(mat1.Zs & mat2.Xs))%2 == 0
# merging Pauli indices (coefficient not determined here)
def pdot(mat1, mat2):
    return Mat(mat1.Xs ^ mat2.Xs, mat1.Zs ^ mat2.Zs)
''' Term: a Mat with coefficient and position
Term.mat :: Mat : matrix of Pauli operator
Term.val :: numeric : coefficient
Term.pos :: int : my position in Ham.terms
'''
class Term:
    def __init__(self, *arg):
        l_arg = len(arg)
        if l_arg == 2:
            self.mat, self.val = arg
        elif l_arg == 1:
            self.mat = arg[0]
            self.val = 1.        
        elif l_arg == 0:
            self.mat = mkMat()
            self.val = 1.
        self.pos = 0
        self._key = None
    def __repr__(self):
        return '%s %s' % (self.val, self.mat)
    def __hash__(self):
        if self._key is None:
            self._key = hash((self.val, self.mat))
        return self._key
    def __eq__(self, other):
        return abs(self.val) == abs(other.val) 
    def __lt__(self, other):
        return abs(self.val) < abs(other.val) 
# dot product of two terms
def dot(term1, term2):
    mat1 = term1.mat
    mat2 = term2.mat
    mat = pdot(mat1, mat2)
    n = mat1.ipower() + mat2.ipower() - mat.ipower()
    n = n + 2*len(mat1.Zs & mat2.Xs)
    s = (-1)**(n/2)
    term = Term(mat, s*term1.val*term2.val)
    return term
# dot product of two terms (times additional i)
def idot(term1, term2):
    mat1 = term1.mat
    mat2 = term2.mat
    mat = pdot(mat1, mat2)
    n = mat1.ipower() + mat2.ipower() - mat.ipower()
    n = n + 2*len(mat1.Zs & mat2.Xs) + 1
    s = (-1)**(n/2)
    return Term(mat, s*term1.val*term2.val)
''' Ham: a collection of Terms
Ham.terms :: list : terms stored in binary heap structure
Ham.mats  :: dict : mapping mat to term
Ham.imap  :: dict : mapping site to covering terms
'''
class Ham:
    def __init__(self, *arg):
        self.terms = []
        self.mats = {}
        self.imap = {}
        if len(arg) == 1:
            self.extend(arg[0])
    def __repr__(self):
        return '%s' % self.terms
    def __len__(self):
        return len(self.terms)
    def __bool__(self):
        return bool(self.terms)
    def __iter__(self):
        return iter(self.terms)
    # add a term to the heap tree (self.terms)
    def terms_push(self, term):
        pos = len(self.terms) # set pos to the end of self.terms
        term.pos = pos
        self.terms.append(term) # append from IR end
        self.terms_shiftUV(pos) # shifted to UV
    # adjust the position of a term in the heap tree
    def terms_adjust(self, term):
        pos = term.pos
        self.terms_shiftUV(pos)
        self.terms_shiftIR(pos)
    # shifting a term indexed by pos in the heap tree towards UV (upward)
    def terms_shiftUV(self, pos):
        terms = self.terms
        this_term = terms[pos]
        # Follow the path to the root, moving parents down until fits.
        while pos > 0:
            parent_pos = (pos - 1) >> 1
            parent_term = terms[parent_pos]
            if abs(this_term.val) > abs(parent_term.val):
                parent_term.pos = pos
                terms[pos] = parent_term
                pos = parent_pos
                continue
            break
        if pos != this_term.pos: # if pos is new
            this_term.pos = pos
            terms[pos] = this_term
    # shifting a term indexed by pos in the heap tree towards IR (downward)
    def terms_shiftIR(self, pos):
        terms = self.terms
        end_pos = len(terms) - 1
        this_term = terms[pos]
        child_pos = 2*pos + 1 # left child position
        while child_pos <= end_pos:
            # Set child_pos to index of larger child.
            rchild_pos = child_pos + 1 # right child position
            if rchild_pos <= end_pos and abs(terms[child_pos].val) < abs(terms[rchild_pos].val):
                child_pos = rchild_pos
            # Move the larger child up.
            child_term = terms[child_pos]
            if abs(this_term.val) < abs(child_term.val):
                child_term.pos = pos
                terms[pos] = child_term
                pos = child_pos
                child_pos = 2*pos + 1 # left child position
                continue
            break
        if pos != this_term.pos: # if pos is new
            this_term.pos = pos
            terms[pos] = this_term
    def imap_add(self, term):
        mat = term.mat
        for i in mat.Xs | mat.Zs:
            try:
                self.imap[i].add(term)
            except:
                self.imap[i] = {term}
    def imap_del(self, term):
        mat = term.mat
        for i in mat.Xs | mat.Zs:
            self.imap[i].remove(term)
    # push a term into the Hamiltonian
    def push(self, term):
        if term.mat in self.mats: # if mat already exist
            old_term = self.mats[term.mat]
            old_term.val += term.val
            self.terms_adjust(old_term)
        else: # if mat is new
            self.terms_push(term)
            self.mats[term.mat] = term
            self.imap_add(term)
    # extend Hamiltonian by adding terms (given by iterator)
    def extend(self, terms):
        for term in terms:
            self.push(term)
    # remove a term from the Hamiltonian
    def remove(self, term):
        terms = self.terms
        end_pos = len(terms) - 1
        pos = term.pos
        del self.mats[term.mat]
        self.imap_del(term)
        if pos == end_pos:
            del terms[pos]
        elif 0 <= pos < end_pos:
            last_term = terms.pop()
            last_term.pos = pos
            terms[pos] = last_term
            self.terms_adjust(last_term)
    # perform C4 rotation generated by sgn*gen to Hamiltonian
    def C4(self, gen, sgn = +1):
        mats = self.mats
        imap = self.imap
        gen_mat = gen.mat
        # collect terms to be transformed
        relevant_terms = set() # start with empty set
        for i in gen_mat.Xs | gen_mat.Zs: # supporting sites of gen
            if i in imap: # if i registered in imap
                relevant_terms.update(imap[i])
        relevant_terms = [term for term in relevant_terms if not is_commute(term.mat, gen_mat)]
        for term in relevant_terms:
            # remove mat
            del mats[term.mat]
            self.imap_del(term)
            # C4 by idot with gen
            new_term = idot(term, gen)
            # update mat & val only
            term.mat = new_term.mat
            term.val = sgn * new_term.val
        # add new mats, NOT COMBINE TO ABOVE LOOP
        for term in relevant_terms:
            mats[term.mat] = term
            self.imap_add(term)
    # perform a series of C4 rotations Rs forward
    def forward(self, Rs):
        for R in Rs:
            self.C4(R)
    # perform a series of C4 rotations Rs backward
    def backward(self, Rs):
        for R in reversed(Rs):
            self.C4(R,-1)
''' Ent: calculate entanglement entropy of stablizers
Ent.mat2is :: dict : mapping from mat to the supporting sites
Ent.i2mats :: dict : mapping from site to the covering mat
Ent.subsys :: set  : entanglement subsystem (a set of sites)
Ent.shared :: set  : a set of mats shared between region and its complement
'''
import numpy as np
#from fortran_ext import z2rank
@jit(nopython=True)
def fast_z2rank(mat):
    # mat input as numpy.matrix, and destroyed on output!
    # caller must ensure mat contains only 0 and 1.
    nr, nc = mat.shape # get num of rows and cols
    r = 0 # current row index
    for i in range(nc): # run through cols
        if r == nr: # row exhausted first
            return r # row rank is full, early return
        if mat[r, i] == 0: # need to find pivot
            found = False # set a flag
            for k in range(r + 1, nr):
                if mat[k, i]: # mat[k, i] nonzero
                    found = True # pivot found in k
                    break
            if found: # if pivot found in k
                #mat[r, :], mat[k, :] = mat[k, :], mat[r, :]
                #
                #tmp = mat[k,:].copy()
                #mat[k,:]=mat[r,:].copy()
                #mat[r,:] = tmp
                #
                for exchange_id in range(0,nc):
                    tmp = mat[k,exchange_id]
                    mat[k,exchange_id] = mat[r, exchange_id]
                    mat[r,exchange_id] = tmp
            else: # if pivot not found
                continue # done with this col
        # pivot has moved to mat[r, i], perform GE
        for j in range(r + 1, nr):
            if mat[j, i]: # mat[j, i] nonzero
                mat[j, i:] = (mat[j, i:] + mat[r, i:])%2
        r = r + 1 # rank inc
    # col exhausted, last nonvanishing row indexed by r
    return r
class Ent:
    def __init__(self, taus):
        self.mat2is = {}
        self.i2mats = {}
        for term in taus:
            mat = term.mat
            sites = mat.Xs | mat.Zs
            self.mat2is[term.mat] = sites
            for i in sites:
                try:
                    self.i2mats[i].add(mat)
                except:
                    self.i2mats[i] = {mat}
        self.clear()
    def is_shared(self, mat):
        sites = self.mat2is[mat]
        return 0 < len(sites & self.subsys) < len(sites)
    def update_shared(self, sites):
        mats = set() # prepare to collect relevant mats
        for i in sites: # scan over relevant sites
            mats.update(self.i2mats[i]) # union into mats
        for mat in mats:
            if self.is_shared(mat): # if shared
                self.shared.add(mat) # add to shared
            else: # if not shared, discard if present in shared
                self.shared.discard(mat)
    # include sites to entanglement region
    def include(self, sites):
        self.subsys.update(sites)
        self.update_shared(sites)
    # exclude sites from entanglement region
    def exclude(self, sites):
        self.subsys.difference_update(sites)
        self.update_shared(sites)
    # clear
    def clear(self):
        self.subsys = set()
        self.shared = set()
    # return entropy of the entanglement region
    def entropy(self):
        mats = [Mat(mat.Xs & self.subsys, mat.Zs & self.subsys) for mat in self.shared]
        # mats is a list of Pauli monomials as generators
        n = len(mats) # get num of projected stablizers
        adj = np.zeros((n, n), dtype=int) # prepare empty adj mat
        # construct adj mat
        for k1 in range(n):
            for k2 in range(k1 + 1, n):
                if not is_commute(mats[k1], mats[k2]):
                    adj[k1, k2] = adj[k2, k1] = 1
        return fast_z2rank(adj)/2
# half-system-size bipartite entropy (averaged over translation)
def bipartite_entropy(system):
    ent = Ent(system.taus)
    l_cut = 0
    L = int(system.size/2)
    S = 0
    ent.include(range(l_cut, l_cut + L))
    for l_cut in range(0, system.size):
        S += ent.entropy()
        ent.exclude({l_cut})
        ent.include({(l_cut + L) % system.size})
    return S/system.size
def subsystem_entropy(system,sub_l):
    ent = Ent(system.taus)
    l_cut = 0
    S = 0
    ent.include(range(l_cut,l_cut+sub_l))
    for l_cut in range(0, system.size):
        S += ent.entropy()
        ent.exclude({l_cut})
        ent.include({(l_cut + sub_l) % system.size})
    return S/system.size
''' SBRG: doing RG, holding RG data and performing data analysis
SBRG.tol      :: float : terms with energy < leading energy * tol will be truncated
SBRG.max_rate :: float : each RG step allows at most (max_rate * num of off-diagonal terms) amount of new terms
SBRG.size     :: int : num of bits in the Hilbert space
SBRG.phybits  :: set : a collection of physical bits
SBRG.H        :: Ham : where the Hamiltonian is held and processed
SBRG.Hbdy     :: list : keep the original terms passed in with the model
SBRG.Hblk     :: list : holographic bulk Hamiltonian transformed by RCC
SBRG.Heff     :: list : terms in the effective Hamiltonian
SBRG.RCC      :: list : C4 transformations from beginning to end
SBRG.taus     :: Ham : stabilizers
SBRG.trash    :: list : hold the energy scales that has been truncated
'''
class SBRG:
    tol = 1.e-1000
    max_rate = 5.
    max_len = 5000
    def __init__(self, model):
        self.size = model.size
        self.phybits = set(range(self.size))
        self.H = Ham(deepcopy(model.terms))
        self.Hbdy = model.terms
        self.Hblk = None
        self.Heff = []
        self.RCC = []
        self.taus = None
        self.trash = []
    def findRs(self, mat):
        if len(mat.Xs) > 0: # if X or Y exists, use it to pivot the rotation
            pbit = min(mat.Xs) # take first off-diag qubit
            return ([idot(Term(mkMat(set(),{pbit})), Term(mat))], pbit)
        else: # if only Z
            if len(mat.Zs) > 1:
                for pbit in sorted(list(mat.Zs)): # find first Z in phybits
                    if (pbit in self.phybits):
                        tmp = Term(mkMat({pbit},set())) # set intermediate term
                        return ([idot(tmp, Term(mat)), idot(Term(mkMat(set(),{pbit})), tmp)], pbit)
            elif len(mat.Zs) == 1:
                pbit = min(mat.Zs)
        return ([], pbit)
    def perturbation(self, H0, offdiag):
        h0 = H0.val # set h0
        min_prod = abs(h0)**2*SBRG.tol # set minimal product

        #try:
            #maxtern = max(offdiag)
            #if abs(maxtern.val/h0) > 0.2:
                #print('step:', self.size - len(self.phybits))
                #print('H0:', H0)
                #print('ratio:', abs(maxtern.val/h0))
                #print('length_off_diag:', len(offdiag))
        #except:
            #print()

        # SiSj for commuting terms whose product val > min_prod
        SiSj = [dot(term1, term2) for (term1, term2) in combinations(offdiag, 2)
                if is_commute(term1.mat,term2.mat) and abs(term1.val*term2.val) > min_prod]
        SiSj.sort() # sort by val
        #print(SiSj)
        # term number truncation
        max_len = min(round(SBRG.max_rate*len(offdiag)), SBRG.max_len)
        if len(SiSj) > max_len:
            print(len(SiSj))
            self.trash.extend([term.val/h0 for term in SiSj[:-max_len]])
            #jhk
            truncation_error = np.sum(np.abs([term.val/h0 for term in SiSj[:-max_len]]))
            if truncation_error > 1e-2:
                print( 'the sum of the absolute values of trash is ', truncation_error )
                print( 'the max_len is {} and the total num is {}'.format(max_len, len(SiSj)) )
            elif truncation_error > 2:
                print( 'the sum of the absolute values of trash is ', truncation_error )
                print( 'the max_len is {} and the total num is {}'.format(max_len, len(SiSj)) )
                print('break!')
                return []
            #jhk

            SiSj = SiSj[-max_len:]
        # multiply by H0 inverse
        H0inv = Term(H0.mat,1/h0)
        pert = [dot(H0inv,term) for term in SiSj]
        # add backward correction
        var = sum((term.val)**2 for term in offdiag) # also used in error estimate
        pert.append(Term(H0.mat, var/(2*h0)))
        return pert
    def nextstep(self):
        if not (self.phybits and self.H): # return if no physical bits or no H
            self.phybits = set() # clear physical bits
            return self
        # get leading energy scale
        H0 = self.H.terms[0]
        h0 = H0.val
        if not abs(h0): # if leading scale vanishes
            self.phybits = set() # quench physical space
            return self
        # find Clifford rotations
        Rs, pbit = self.findRs(H0.mat)
        self.RCC.extend(Rs) # add to RCC
        self.H.forward(Rs) # apply to H
        # pick out offdiag terms
        offdiag = [term for term in self.H.imap[pbit] if pbit in term.mat.Xs]
        pert = self.perturbation(H0, offdiag) # 2nd order perturbation
        for term in offdiag:
            self.H.remove(term) # remove off-diagonal terms
        self.H.extend(pert) # add perturbation to H
        self.phybits.remove(pbit) # reduce physical bits
        # remove identity terms in physical space
        for term in list(self.H.imap[pbit]): # NOT REMOVE list(...)
            if not ((term.mat.Xs | term.mat.Zs) & self.phybits):
                self.Heff.append(term)
                self.H.remove(term)
        return (Term(H0.mat,h0), Rs, offdiag)
    def flow(self, step = float('inf')):
        step = min(step, len(self.phybits)) # adjust RG steps
        # carry out RG flow
        stp_count = 0
        while self.phybits and stp_count < step:
            self.nextstep()
            stp_count += 1
    def make(self):
        # reconstruct stabilizers
        stabilizers = []
        blkbits = set(range(self.size))
        for term in self.Heff:
            if len(term.mat.Zs) == 1:
                stabilizers.append(deepcopy(term))
                blkbits -= term.mat.Zs
        self.taus_unc4 = deepcopy(stabilizers)
        stabilizers.extend(Term(mkMat(set(),{i}),0) for i in blkbits)
        self.taus = Ham(stabilizers)
        self.taus.backward(self.RCC)
        # reconstruct holographic bulk Hamiltonian
        self.Hblk = Ham(deepcopy(self.Hbdy))
        self.Hblk.forward(self.RCC)
    def run(self):
        self.flow()
        self.make()
        return self
    # calculate Anderson correlator between pairs in terms
    def correlate(self, terms):
        ops = Ham(terms)
        ops.forward(self.RCC)
        cor = {}
        L = self.size
        for (i,j) in combinations(range(len(ops)),2):
            if len(ops.terms[i].mat.Xs ^ ops.terms[j].mat.Xs) == 0:
                d = int(abs((j - i + L/2)%L - L/2))
                cor[d] = cor.get(d,0) + 1
        return cor
    def site2Heffmats(self):
        self.s2Heff = {}
        for terms in self.Heff:
            for site in terms.mat.Zs:
                try:
                    self.s2Heff[site].append(terms)
                except:
                    self.s2Heff[site] = [terms]
    def energy(self, state):
        gs_energy = 0
        for term in self.Heff:
            opt_bits = [state[i] for i in term.mat.Zs]
            gs_energy += np.prod(opt_bits) * term.val
        return gs_energy
    # note that the multi-Zs terms from perturbation are much smaller than the single-Zs terms
    def grndstate_blk(self):
        ground_state = [-1] * self.size
        blkbits = set(range(self.size))
        for term in self.Heff:
            if len(term.mat.Zs) == 1:
                site = list(term.mat.Zs)[0]
                ground_state[site] = 1-2*(term.val>0)
                blkbits -= term.mat.Zs
        #print(blkbits)
        self.Heff.extend(Term(mkMat(set(),{i}),0) for i in blkbits)                
        gs_energy = self.energy(ground_state)
        return ground_state, gs_energy
    # dynamical spin-spin struc coherent factor: S(i,j,\omega)==c_ij*f(omega) (ground state)
    def two_spin_chf(self, ops, ground_state):
        energy0 = self.energy(ground_state)
        ops.forward(self.RCC)
        struc_coef = {}
        for i in range(len(ops)):
            for j in range(i, len(ops)):
                termi = ops.terms[i]
                termj = ops.terms[j]
                if len(termi.mat.Xs ^ termj.mat.Xs) == 0:
                    state1 = ground_state.copy()
                    phase = termi.val;
                    for site in termi.mat.Zs:
                        phase *= state1[site]
                    for site in termi.mat.Xs:
                        state1[site] *= -1
                    phase *= (1j)**(termi.mat.ipower())
                    energy1 = self.energy(state1)
                    phase *= termj.val;
                    for site in termj.mat.Zs:
                        phase *= state1[site]
                    #for site in termj.mat.Xs:
                        #state1[site] *= -1
                    phase *= (1j)**(termj.mat.ipower())
                    struc_coef[(i,j)] = (phase.real, energy1-energy0)
                else:
                    struc_coef[(i,j)] = ()
        return struc_coef
    def two_spin_chf2(self, ops, ground_state):
        ops.forward(self.RCC)
        struc_coef = {}
        if not 's2Heff' in self.__dir__():
            self.site2Heffmats()
        for i in range(len(ops)):
            for j in range(len(ops)):
                termi = ops.terms[i]
                termj = ops.terms[j]
                if len(termi.mat.Xs ^ termj.mat.Xs) == 0:
                    phase = termi.val;
                    for site in termi.mat.Zs:
                        phase *= ground_state[site]
                    energy_difference = 0 
                    for site in termi.mat.Xs:
                        ground_state[site] *= -1
                        for term in self.s2Heff[site]:
                            opt_bits = [ground_state[ss] for ss in term.mat.Zs]
                            energy_difference += 2 * np.prod(opt_bits) * term.val
                    phase *= (1j)**(termi.mat.ipower())
                    phase *= termj.val;
                    for site in termj.mat.Zs:
                        phase *= ground_state[site]
                    for site in termj.mat.Xs:
                        ground_state[site] *= -1
                    phase *= (1j)**(termj.mat.ipower())
                    struc_coef[str(i)+"-"+str(j)] = (phase.real, energy_difference)
                else:
                    struc_coef[str(i)+"-"+str(j)] = ()
        return struc_coef
    def two_spin_chf3(self, ops, ground_state):
        ops.forward(self.RCC)
        struc_coef = {}
        if not 's2Heff' in self.__dir__():
            self.site2Heffmats()
        for i in range(len(ops)):
            for j in range(len(ops)):
                termi = ops.terms[i]
                termj = ops.terms[j]
                if len(termi.mat.Xs ^ termj.mat.Xs) == 0:
                    phase = termi.val;
                    for site in termi.mat.Zs:
                        phase *= ground_state[site]
                    energy_difference = 0 
                    for site in termi.mat.Xs:
                        ground_state[site] *= -1
                        for term in self.s2Heff[site]:
                            opt_bits = [ground_state[ss] for ss in term.mat.Zs]
                            energy_difference += 2 * np.prod(opt_bits) * term.val
                    phase *= (1j)**(termi.mat.ipower())
                    phase *= termj.val;
                    for site in termj.mat.Zs:
                        phase *= ground_state[site]
                    for site in termj.mat.Xs:
                        ground_state[site] *= -1
                    phase *= (1j)**(termj.mat.ipower())
                    struc_coef[(i,j)] = (phase.real, energy_difference)
                else:
                    struc_coef[(i,j)] = ()
        return struc_coef
    # return <state|op|state> for op in ops
    def measure_ops(self, ops, state):
        ops.forward(self.RCC)
        msrmnts = []
        for i in range(len(ops)):
            termi = ops.terms[i]            
            if len(termi.mat.Xs) == 0:
                phase = termi.val;
                for site in termi.mat.Zs:
                    phase *= state[site]
                msrmnts.append( phase )
            else:
                msrmnts.append( 0 )
        return msrmnts
    # dynamical spin-spin correlation function  S(i,j,\omega) (ground state)    
    def two_spin_correlation(self, struc_chf, omega, delta=0.01):
        # use the lorentz line shape to fit delta function
        spectrum = 0;
        for n in range(self.size):
            for m in range(self.size):
                if len(struc_chf[str(n)+"-"+str(m)])!=0:
                # if not isinstance(struc_chf[(n,m)], int):
                    spectrum += struc_chf[str(n)+"-"+str(m)][0] * lorentz_line(omega, struc_chf[str(n)+"-"+str(m)][1], delta)
        return spectrum/self.size*2

    def Sqw(self, struc_chf, q2ijs, omegas, delta,use_box):
        '''<input>: struc_chf : dict, see two_spin_chf(2) 
                omegas: 1D array
                q2ijs : list with dict as elements
                delta: float
        '''
        if isinstance(q2ijs,list):
            spectrum = np.zeros( (len(omegas), len(q2ijs)))
            for _, q in enumerate(q2ijs):
                for i in range(self.size):
                    for j in range(self.size):
                        key = str(i)+"-"+str(j)
                        if not isinstance(struc_chf[key], int):
                            try:
                                if use_box == 1:
                                    spectrum[:, _] += (lorentz_line(omegas, struc_chf[key][1], delta) * struc_chf[key][0] * (np.exp(1j*q[key])).real )
                                else:
                                    tmp_delta_omega = omegas[1]-omegas[0]
                                    tmp_bin = np.linspace(0,omegas[-1]+tmp_delta_omega,np.shape(omegas)[0]+1)-0.5*tmp_delta_omega
                                    tmp_hist, bins = np.histogram(struc_chf[key][1],bins=tmp_bin)
                                    spectrum[:, _] += (tmp_hist * struc_chf[key][0] * (np.exp(1j*q[key])).real )
                            except:
                                continue
                        #print(np.cos(q2ij[key]))
            return spectrum#/self.size
        elif q2ijs == 0:
            # S(q=0, w)
            spectrum = 0
            for n in range(self.size):
                for m in range(self.size):
                    key = str(n)+"-"+str(m)
                    if not isinstance(struc_chf[key], int):
                        spectrum += struc_chf[key][0] * lorentz_line(omegas, struc_chf[key][1], delta)
            return spectrum#/self.size
        else:
            print('q2ijs should be a list of dict or zero')
            return 
    def Sqw_nn(self, struc_chf, q2ijs, omegas, delta,use_box,lx,ly):
        '''<input>: struc_chf : dict, see two_spin_chf(2) 
                omegas: 1D array
                q2ijs : list with dict as elements
                delta: float
        '''
        if isinstance(q2ijs,list):
            spectrum = np.zeros( (len(omegas), len(q2ijs)))
            for _, q in enumerate(q2ijs):
                for i in range(self.size):
                    for j in range(self.size):
                        ix = i%lx
                        iy = i//lx
                        jx = j%lx
                        jy = j//lx
                        x_diff = jx-ix
                        y_diff = jy-iy
                        dist = np.sqrt((x_diff+0.5*y_diff)**2+(np.sqrt(3)*y_diff/2.)**2)
                        key = str(i)+"-"+str(j)
                        if not isinstance(struc_chf[key], int):
                            try:
                                if use_box == 1:
                                    spectrum[:, _] += (lorentz_line(omegas, struc_chf[key][1], delta) * struc_chf[key][0] * (np.exp(1j*q[key])).real )
                                else:
                                    tmp_delta_omega = omegas[1]-omegas[0]
                                    tmp_bin = np.linspace(0,omegas[-1]+tmp_delta_omega,np.shape(omegas)[0]+1)-0.5*tmp_delta_omega
                                    tmp_hist, bins = np.histogram(struc_chf[key][1],bins=tmp_bin)
                                    if dist <2.:
                                        spectrum[:, _] += (tmp_hist * struc_chf[key][0] * (np.exp(1j*q[key])).real )
                                    else:
                                        spectrum[:, _] += (tmp_hist * 0. * (np.exp(1j*q[key])).real )
                            except:
                                continue
                        #print(np.cos(q2ij[key]))
            return spectrum#/self.size
        elif q2ijs == 0:
            # S(q=0, w)
            spectrum = 0
            for n in range(self.size):
                for m in range(self.size):
                    key = str(n)+"-"+str(m)
                    if not isinstance(struc_chf[key], int):
                        spectrum += struc_chf[key][0] * lorentz_line(omegas, struc_chf[key][1], delta)
            return spectrum#/self.size
        else:
            print('q2ijs should be a list of dict or zero')
            return 
                       
    
        
''' Model: defines Hilbert space and Hamiltonian
Model.size  :: int : num of bits
Model.terms :: list : terms in the Hamiltonian
'''
class Model:
    def __init__(self):
        self.size = 0
        self.terms = []
'''the beta function from uniform distribution'''

def rnd_beta(alpha):
    return np.random.rand()**(1./alpha) if alpha > 0 else 1

# quantum Ising model
def TFIsing(L, **para):
    # L - number of sites (assuming PBC)
    # model - a dict of model parameters
    try: # set parameter alpha
        alpha = para['alpha']
        alpha_J = alpha
        alpha_K = alpha
        alpha_h = alpha
    except:
        alpha_J = para.get('alpha_J',1)
        alpha_K = para.get('alpha_K',1)
        alpha_h = para.get('alpha_h',1)
    model = Model()
    model.size = L
    # translate over the lattice by deque rotation
    H_append = model.terms.append
    rnd_beta = random.betavariate
    for i in range(L):
        H_append(Term(mkMat({i: 1, (i+1)%L: 1}), para['J']*rnd_beta(alpha_J, 1)))
        H_append(Term(mkMat({i: 3, (i+1)%L: 3}), para['K']*rnd_beta(alpha_K, 1)))
        H_append(Term(mkMat({i: 3}), para['h']*rnd_beta(alpha_h, 1)))
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model
# XYZ model
def XYZ(L, **para):
    # L - number of sites (assuming PBC)
    # model - a dict of model parameters
    try: # set parameter alpha
        alpha = para['alpha']
        alpha_X = alpha
        alpha_Y = alpha
        alpha_Z = alpha
    except:
        alpha_X = para.get('alpha_x',1)
        alpha_Y = para.get('alpha_y',1)
        alpha_Z = para.get('alpha_z',1)
    model = Model()
    model.size = L
    # translate over the lattice by deque rotation
    H_append = model.terms.append
    for i in range(L):
        H_append(Term(mkMat({i: 1, (i+1)%L: 1}), para['Jx']*rnd_beta(alpha_X)))
        H_append(Term(mkMat({i: 2, (i+1)%L: 2}), para['Jy']*rnd_beta(alpha_Y)))
        H_append(Term(mkMat({i: 3, (i+1)%L: 3}), para['Jz']*rnd_beta(alpha_Z)))
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model
# Toric code model
def toric_code(lx, ly, **para):
    # lx - lattice size in x direction
    # ly - lattice size in y direction
    try: # set parameter alpha
        alpha_Jx = para['alpha']
        alpha_Jz = para['alpha']
        alpha_hx = para['alpha']
        alpha_hz = para['alpha']
        gamma_Jx = para['gamma']
        gamma_Jz = para['gamma']
        gamma_hx = para['gamma']
        gamma_hz = para['gamma']
    except:
        alpha_Jx = para.get('alpha_jx',1)
        alpha_Jz = para.get('alpha_jz',1)
        alpha_hx = para.get('alpha_hx',1)
        alpha_hz = para.get('alpha_hz',1)
        gamma_Jx = para.get('gamma_jx',1)
        gamma_Jz = para.get('gamma_jz',1)
        gamma_hx = para.get('gamma_hx',1)
        gamma_hz = para.get('gamma_hz',1)
    model = Model()
    model.size = lx*ly*2
    model.lx = lx
    model.ly = ly
    n = lx*ly
    H_append = model.terms.append
    for ix in range(lx):
        for iy in range(ly):
            site_row = ix + iy*lx
            site_col = ix + iy*lx + n
            site_row_mx = (ix-1)%lx + iy*lx
            site_col_my = ix + ((iy-1)%ly)*lx + n
            site_row_py = ix + ((iy+1)%ly)*lx
            site_col_px = (ix+1)%lx + iy*lx + n
            H_append( Term(mkMat({site_row: 1, site_col: 1, site_row_mx: 1, site_col_my: 1}),
                           para['jx']*rnd_beta(alpha_Jx)) )
            H_append( Term(mkMat({site_row: 3, site_col: 3, site_row_py: 3, site_col_px: 3}),
                           para['jz']*rnd_beta(alpha_Jz)) )
            H_append( Term(mkMat({site_row: 1}), para['hx']*rnd_beta(alpha_hx)) )
            H_append( Term(mkMat({site_col: 1}), para['hx']*rnd_beta(alpha_hx)) )
            H_append( Term(mkMat({site_row: 3}), para['hz']*rnd_beta(alpha_hz)) )
            H_append( Term(mkMat({site_col: 3}), para['hz']*rnd_beta(alpha_hz)) )
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model
#Kitaev honey combo model
def Kitaev(lx, ly, **para):
    # lx - lattice size in x direction
    # ly - lattice size in y direction
    try: # set parameter alpha
        alpha = para['alpha']
        alpha_Jx = alpha
        alpha_Jy = alpha
        alpha_Jz = alpha
    except:
        alpha_Jx = para.get('alpha_jx',1)
        alpha_Jy = para.get('alpha_jy',1)
        alpha_Jz = para.get('alpha_jz',1)
    model = Model()
    model.size = lx*ly*2
    model.n = lx*ly
    model.lx = lx
    model.ly = ly
    n = lx*ly
    H_append = model.terms.append
    #rnd_beta = random.betavariate
    for ix in range(lx):
        for iy in range(ly):
            site_A = ix + iy*lx
            site_B = ix + iy*lx + n
            site_B_mx = (ix-1)%lx + (iy)*lx + n
            site_B_my = ix + ((iy-1)%ly)*lx + n
            H_append( Term(mkMat({site_A: 1, site_B: 1}),
                           para['jx']*rnd_beta(alpha_Jx)) )
            H_append( Term(mkMat({site_A: 2, site_B_mx: 2}),
                           para['jy']*rnd_beta(alpha_Jy)) )
            H_append( Term(mkMat({site_A: 3, site_B_my: 3}),
                           para['jz']*rnd_beta(alpha_Jz)) )
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model

def triangular_XYZ(Lx,Ly, **para):
    # assuming PBC 
    alpha = para['alpha']
    alpha_X = alpha
    alpha_Y = alpha
    alpha_Z = alpha
    model = Model()
    model.size = Lx*Ly
    model.lx = Lx
    model.ly = Ly
    H_append = model.terms.append
    rand_uni = np.random.uniform
    coor_to_id = lambda x,y: y*Lx + x
    for i in range(Lx):
        for j in range(Ly):
            coef_Jx = rnd_beta(alpha_X)*para['jx']
            coef_Jy = rnd_beta(alpha_Y)*para['jy']
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term( mkMat({coor_to_id(i,j):1, coor_to_id((i+1)%Lx,j):1}),coef_Jx ))
            H_append(Term( mkMat({coor_to_id(i,j):2, coor_to_id((i+1)%Lx,j):2}),coef_Jy ))
            H_append(Term( mkMat({coor_to_id(i,j):3, coor_to_id((i+1)%Lx,j):3}),coef_Jz ))
            coef_Jx = rnd_beta(alpha_X)*para['jx']
            coef_Jy = rnd_beta(alpha_Y)*para['jy']
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term( mkMat({coor_to_id(i,j):1, coor_to_id(i,(j+1)%Ly):1}),coef_Jx ))
            H_append(Term( mkMat({coor_to_id(i,j):2, coor_to_id(i,(j+1)%Ly):2}),coef_Jy ))
            H_append(Term( mkMat({coor_to_id(i,j):3, coor_to_id(i,(j+1)%Ly):3}),coef_Jz ))
            coef_Jx = rnd_beta(alpha_X)*para['jx']
            coef_Jy = rnd_beta(alpha_Y)*para['jy']
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term( mkMat({coor_to_id(i,j):1, coor_to_id((i-1)%Lx, (j+1)%Ly):1}),coef_Jx ))
            H_append(Term( mkMat({coor_to_id(i,j):2, coor_to_id((i-1)%Lx, (j+1)%Ly):2}),coef_Jy ))
            H_append(Term( mkMat({coor_to_id(i,j):3, coor_to_id((i-1)%Lx, (j+1)%Ly):3}),coef_Jz ))
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model   
def cylinder_triangular_XYZ(Lx, Ly, **para):
    alpha = para['alpha']
    alpha_X = alpha
    alpha_Y = alpha
    alpha_Z = alpha
    model = Model()
    model.size = Lx*Ly*2
    model.lx = Lx
    model.ly = Ly
    H_append = model.terms.append
    def ijmu2id(i,j,mu):
        return 2.*(j*Ly+i)+mu
    def id2ijmu(idx):
        tmp_mu = idx%2
        tmp_i = ((idx-tmp_mu)/2)%Lx
        tmp_j = ((idx-tmp_mu)/2-tmp_i)/Lx
        return tmp_i,tmp_j_tmp_mu
    for i in range(Lx):
        for j in range(Ly):
            #1
            coef_Jx = rnd_beta(alpha_X)*para['jx']
            coef_Jy = rnd_beta(alpha_Y)*para['jy']
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term( mkMat({ijmu2id(i,j,0):1, ijmu2id((i-1)%Lx,(j-1)%Ly,1):1}),coef_Jx))
            H_append(Term( mkMat({ijmu2id(i,j,0):2, ijmu2id((i-1)%Lx,(j-1)%Ly,1):2}),coef_Jy ))
            H_append(Term( mkMat({ijmu2id(i,j,0):3, ijmu2id((i-1)%Lx,(j-1)%Ly,1):3}),coef_Jz ))
            #2
            coef_Jx = rnd_beta(alpha_X)*para['jx']
            coef_Jy = rnd_beta(alpha_Y)*para['jy']
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term( mkMat({ijmu2id(i,j,0):1, ijmu2id((i-1)%Lx,j,0):1}),coef_Jx))
            H_append(Term( mkMat({ijmu2id(i,j,0):2, ijmu2id((i-1)%Lx,j,0):2}),coef_Jy ))
            H_append(Term( mkMat({ijmu2id(i,j,0):3, ijmu2id((i-1)%Lx,j,0):3}),coef_Jz ))
            #3
            coef_Jx = rnd_beta(alpha_X)*para['jx']
            coef_Jy = rnd_beta(alpha_Y)*para['jy']
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term( mkMat({ijmu2id(i,j,0):1, ijmu2id((i-1)%Lx,j,1):1}),coef_Jx))
            H_append(Term( mkMat({ijmu2id(i,j,0):2, ijmu2id((i-1)%Lx,j,1):2}),coef_Jy ))
            H_append(Term( mkMat({ijmu2id(i,j,0):3, ijmu2id((i-1)%Lx,j,1):3}),coef_Jz ))
            #4
            coef_Jx = rnd_beta(alpha_X)*para['jx']
            coef_Jy = rnd_beta(alpha_Y)*para['jy']
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term( mkMat({ijmu2id(i,j,0):1, ijmu2id(i,(j)%Ly,1):1}),coef_Jx))
            H_append(Term( mkMat({ijmu2id(i,j,0):2, ijmu2id(i,(j)%Ly,1):2}),coef_Jy ))
            H_append(Term( mkMat({ijmu2id(i,j,0):3, ijmu2id(i,(j)%Ly,1):3}),coef_Jz ))
            #5
            coef_Jx = rnd_beta(alpha_X)*para['jx']
            coef_Jy = rnd_beta(alpha_Y)*para['jy']
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term( mkMat({ijmu2id(i,j,1):1, ijmu2id((i-1)%Lx,j,1):1}),coef_Jx))
            H_append(Term( mkMat({ijmu2id(i,j,1):2, ijmu2id((i-1)%Lx,j,1):2}),coef_Jy ))
            H_append(Term( mkMat({ijmu2id(i,j,1):3, ijmu2id((i-1)%Lx,j,1):3}),coef_Jz ))
            #6
            coef_Jx = rnd_beta(alpha_X)*para['jx']
            coef_Jy = rnd_beta(alpha_Y)*para['jy']
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term( mkMat({ijmu2id(i,j,1):1, ijmu2id(i,(j+1)%Ly,0):1}),coef_Jx))
            H_append(Term( mkMat({ijmu2id(i,j,1):2, ijmu2id(i,(j+1)%Ly,0):2}),coef_Jy ))
            H_append(Term( mkMat({ijmu2id(i,j,1):3, ijmu2id(i,(j+1)%Ly,0):3}),coef_Jz ))
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model

def square_XYZ(Lx,Ly, **para):
    # assuming PBC 
    alpha = para['alpha']
    alpha_X = alpha
    alpha_Y = alpha
    alpha_Z = alpha
    model = Model()
    model.size = Lx*Ly
    model.ly = Ly
    model.lx = Lx
    H_append = model.terms.append
    rand_uni = np.random.uniform
    coor_to_id = lambda x,y: y*Lx + x
    for i in range(Lx):
        for j in range(Ly):
            coef_Jx = rnd_beta(alpha_X)*para['jx']
            coef_Jy = rnd_beta(alpha_Y)*para['jy']
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term( mkMat({coor_to_id(i,j):1, coor_to_id((i+1)%Lx,j):1}),coef_Jx ))
            H_append(Term( mkMat({coor_to_id(i,j):2, coor_to_id((i+1)%Lx,j):2}),coef_Jy ))
            H_append(Term( mkMat({coor_to_id(i,j):3, coor_to_id((i+1)%Lx,j):3}),coef_Jz ))
            coef_Jx = rnd_beta(alpha_X)*para['jx']
            coef_Jy = rnd_beta(alpha_Y)*para['jy']
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term( mkMat({coor_to_id(i,j):1, coor_to_id(i,(j+1)%Ly):1}),coef_Jx ))
            H_append(Term( mkMat({coor_to_id(i,j):2, coor_to_id(i,(j+1)%Ly):2}),coef_Jy ))
            H_append(Term( mkMat({coor_to_id(i,j):3, coor_to_id(i,(j+1)%Ly):3}),coef_Jz ))
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model    

def sparse_triangular_XYZ(Lx, Ly, **para):
    eps = 1e-10
    alpha = para['alpha']
    alpha_X = alpha
    alpha_Y = alpha
    alpha_Z = alpha
    model = Model()
    model.size = Lx*Ly
    model.lx = Lx
    model.ly = Ly
    H_append = model.terms.append
    rand_uni = np.random.uniform
    coor_to_id = lambda x,y: y*Lx + x
    id2j = {1:'jx', 2:'jy', 3:'jz'}
    jx_para = para['jx']
    jy_para = para['jy']
    jz_para = para['jz']
    #if (jx_para!=0)&(jy_para==0)&(jz_para==0):
    if (abs(jx_para)>=eps)&(abs(jy_para)<eps)&(abs(jz_para)<eps):
        for i in range(Lx):
            for j in range(Ly):
                xyz = np.random.randint(1,2)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i+1)%Lx,j):xyz}),coef_j ))
                xyz = np.random.randint(1,2)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id(i,(j+1)%Ly):xyz}),coef_j ))
                xyz = np.random.randint(1,2)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i-1)%Lx, (j+1)%Ly):xyz}),coef_j ))
    #elif (jx_para==0)&(jy_para!=0)&(jz_para==0):
    elif (abs(jx_para)<eps)&(abs(jy_para)>=eps)&(abs(jz_para)<eps):
        for i in range(Lx):
            for j in range(Ly):
                xyz = np.random.randint(2,3)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i+1)%Lx,j):xyz}),coef_j ))
                xyz = np.random.randint(2,3)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id(i,(j+1)%Ly):xyz}),coef_j ))
                xyz = np.random.randint(2,3)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i-1)%Lx, (j+1)%Ly):xyz}),coef_j ))
    #elif (jx_para==0)&(jy_para==0)&(jz_para!=0):
    elif (abs(jx_para)<eps)&(abs(jy_para)<eps)&(abs(jz_para)>=eps):
        for i in range(Lx):
            for j in range(Ly):
                xyz = np.random.randint(3,4)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i+1)%Lx,j):xyz}),coef_j ))
                xyz = np.random.randint(3,4)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id(i,(j+1)%Ly):xyz}),coef_j ))
                xyz = np.random.randint(3,4)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i-1)%Lx, (j+1)%Ly):xyz}),coef_j ))
    #elif (jx_para!=0)&(jy_para!=0)&(jz_para==0):
    elif (abs(jx_para)>=eps)&(abs(jy_para)>=eps)&(abs(jz_para)<eps):
        for i in range(Lx):
            for j in range(Ly):
                xyz = np.random.randint(1,3)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i+1)%Lx,j):xyz}),coef_j ))
                xyz = np.random.randint(1,3)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id(i,(j+1)%Ly):xyz}),coef_j ))
                xyz = np.random.randint(1,3)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i-1)%Lx, (j+1)%Ly):xyz}),coef_j ))
    #elif (jx_para!=0)&(jy_para==0)&(jz_para!=0):
    elif (abs(jx_para)>=eps)&(abs(jy_para)<eps)&(abs(jz_para)>=eps):
        for i in range(Lx):
            for j in range(Ly):
                xyz = np.random.randint(1,3)
                if xyz > 1.5:
                    xyz = int(3)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i+1)%Lx,j):xyz}),coef_j ))
                xyz = np.random.randint(1,3)
                if xyz > 1.5:
                    xyz = int(3)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id(i,(j+1)%Ly):xyz}),coef_j ))
                xyz = np.random.randint(1,3)
                if xyz > 1.5:
                    xyz = int(3)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i-1)%Lx, (j+1)%Ly):xyz}),coef_j ))
    #elif (jx_para==0)&(jy_para!=0)&(jz_para!=0):
    elif (abs(jx_para)<eps)&(abs(jy_para)>=eps)&(abs(jz_para)>=eps):
        for i in range(Lx):
            for j in range(Ly):
                xyz = np.random.randint(2,4)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i+1)%Lx,j):xyz}),coef_j ))
                xyz = np.random.randint(2,4)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id(i,(j+1)%Ly):xyz}),coef_j ))
                xyz = np.random.randint(2,4)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i-1)%Lx, (j+1)%Ly):xyz}),coef_j ))
    elif (abs(jx_para)>=eps)&(abs(jy_para)>=eps)&(abs(jz_para)>=eps):
        for i in range(Lx):
            for j in range(Ly):
                xyz = np.random.randint(1,4)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i+1)%Lx,j):xyz}),coef_j ))
                xyz = np.random.randint(1,4)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id(i,(j+1)%Ly):xyz}),coef_j ))
                xyz = np.random.randint(1,4)
                coef_j = rnd_beta(alpha)*para[id2j[xyz]]
                H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i-1)%Lx, (j+1)%Ly):xyz}),coef_j ))
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model
def trilattice_sparse_triangular_XYZ(Lx, Ly, **para):
    alpha = para['alpha']
    alpha_X = alpha
    alpha_Y = alpha
    alpha_Z = alpha
    model = Model()
    model.size = Lx*Ly
    model.lx = Lx
    model.ly = Ly
    H_append = model.terms.append
    rand_uni = np.random.uniform
    coor_to_id = lambda x,y: y*Lx + x
    id2j = {1:'jx', 2:'jy', 3:'jz'}
    for i in range(Lx):
        for j in range(Ly):
            xyz = np.random.randint(1,4)
            coef_j = rnd_beta(alpha)*para[id2j[xyz]]
            H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i+1)%Lx,j):xyz}),coef_j ))
            xyz = np.random.randint(1,4)
            coef_j = rnd_beta(alpha)*para[id2j[xyz]]
            H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id(i,(j+1)%Ly):xyz}),coef_j ))
            xyz = np.random.randint(1,4)
            coef_j = rnd_beta(alpha)*para[id2j[xyz]]
            H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i-1)%Lx, (j+1)%Ly):3}),coef_j ))
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model
def J1J2_trilattice_sparse_triangular_XYZ(Lx, Ly, **para):
    alpha = para['alpha']
    alpha_X = alpha
    alpha_Y = alpha
    alpha_Z = alpha
    model = Model()
    model.size = Lx*Ly
    model.lx = Lx
    model.ly = Ly
    H_append = model.terms.append
    rand_uni = np.random.uniform
    coor_to_id = lambda x,y: y*Lx + x
    id2j1 = {1:'j1x', 2:'j1y', 3:'j1z'}
    id2j2 = {1:'j2x', 2:'j2y', 3:'j2z'}
    for i in range(Lx):
        for j in range(Ly):
            xyz = np.random.randint(1,4)
            coef_j = rnd_beta(alpha)*para[id2j1[xyz]]
            H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i+1)%Lx,j):xyz}),coef_j ))
            xyz = np.random.randint(1,4)
            coef_j = rnd_beta(alpha)*para[id2j1[xyz]]
            H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id(i,(j+1)%Ly):xyz}),coef_j ))
            xyz = np.random.randint(1,4)
            coef_j = rnd_beta(alpha)*para[id2j1[xyz]]
            H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i-1)%Lx, (j+1)%Ly):3}),coef_j ))
            xyz = np.random.randint(1,4)
            coef_j = rnd_beta(alpha)*para[id2j2[xyz]]
            H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i+1)%Lx, (j+1)%Ly):3}),coef_j ))
            xyz = np.random.randint(1,4)
            coef_j = rnd_beta(alpha)*para[id2j2[xyz]]
            H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i-1)%Lx, (j+2)%Ly):3}),coef_j ))
            xyz = np.random.randint(1,4)
            coef_j = rnd_beta(alpha)*para[id2j2[xyz]]
            H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i-2)%Lx, (j+1)%Ly):3}),coef_j ))


    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model
def arg1_trilattice_sparse_triangular_XYZ(Lx, Ly, **para):
    alpha = para['alpha']
    alpha_X = alpha
    alpha_Y = alpha
    alpha_Z = alpha
    model = Model()
    model.size = Lx*Ly
    model.lx = Lx
    model.ly = Ly
    H_append = model.terms.append
    rand_uni = np.random.uniform
    coor_to_id = lambda x,y: y*Lx + x
    id2j = {1:'jx', 2:'jy', 3:'jz'}
    for i in range(Lx):
        for j in range(Ly):
            xyz = np.random.randint(1,4)
            coef_j = rnd_beta(alpha)*para[id2j[xyz]]
            H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i+1)%Lx,j):xyz}),coef_j ))
            xyz = np.random.randint(1,4)
            coef_j = rnd_beta(alpha)*para[id2j[xyz]]
            H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id(i,(j+1)%Ly):xyz}),coef_j ))
            xyz = np.random.randint(1,4)
            coef_j = rnd_beta(alpha)*para[id2j[xyz]]
            H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i-1)%Lx, (j+1)%Ly):3}),coef_j ))
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model
def arg2_trilattice_sparse_triangular_XYZ(Lx, Ly, **para):
    alpha = para['alpha']
    alpha_X = alpha
    alpha_Y = alpha
    alpha_Z = alpha
    model = Model()
    model.size = Lx*Ly
    model.lx = Lx
    model.ly = Ly
    H_append = model.terms.append
    rand_uni = np.random.uniform
    coor_to_id = lambda x,y: y*Lx + x
    id2j = {1:'jx', 2:'jy', 3:'jz'}
    for i in range(Lx):
        for j in range(Ly):
            xyz = np.random.randint(1,4)
            coef_j = rnd_beta(alpha)*para[id2j[xyz]]
            H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i+1)%Lx,j):xyz}),coef_j ))
            xyz = np.random.randint(1,4)
            coef_j = rnd_beta(alpha)*para[id2j[xyz]]
            H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id(i,(j+1)%Ly):xyz}),coef_j ))
            xyz = np.random.randint(1,4)
            coef_j = rnd_beta(alpha)*para[id2j[xyz]]
            H_append(Term( mkMat({coor_to_id(i,j):xyz, coor_to_id((i-1)%Lx, (j+1)%Ly):3}),coef_j ))
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model
def honeycomb_XYZ(Lx, Ly, **para):
    # assuming PBC
    alpha = para['alpha']
    alpha_X = alpha
    alpha_Y = alpha
    alpha_Z = alpha
    model = Model()
    model.size = Lx*Ly*2
    model.ly = Ly
    model.lx = Lx
    H_append = model.terms.append
    #rand_uni = np.random.uniform
    coor_to_id = lambda x, y, mu: (y*Lx + x)*2 + mu
    for i in range(Lx):
        for j in range(Ly):
            coef_Jx = rnd_beta(alpha_X)*para['jx']
            coef_Jy = rnd_beta(alpha_Y)*para['jy']
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term( mkMat({coor_to_id(i,j,0):1, coor_to_id(i,j,1):1}),coef_Jx ))
            H_append(Term( mkMat({coor_to_id(i,j,0):2, coor_to_id(i,j,1):2}),coef_Jy ))
            H_append(Term( mkMat({coor_to_id(i,j,0):3, coor_to_id(i,j,1):3}),coef_Jz ))
            coef_Jx = rnd_beta(alpha_X)*para['jx']
            coef_Jy = rnd_beta(alpha_Y)*para['jy']
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term( mkMat({coor_to_id(i,j,0):1, coor_to_id(i,(j-1)%Ly,1):1}),coef_Jx ))
            H_append(Term( mkMat({coor_to_id(i,j,0):2, coor_to_id(i,(j-1)%Ly,1):2}),coef_Jy ))
            H_append(Term( mkMat({coor_to_id(i,j,0):3, coor_to_id(i,(j-1)%Ly,1):3}),coef_Jz ))
            coef_Jx = rnd_beta(alpha_X)*para['jx']
            coef_Jy = rnd_beta(alpha_Y)*para['jy']
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term( mkMat({coor_to_id(i,j,0):1, coor_to_id((i-1)%Lx,(j-1)%Ly,1):1}),coef_Jx ))
            H_append(Term( mkMat({coor_to_id(i,j,0):2, coor_to_id((i-1)%Lx,(j-1)%Ly,1):2}),coef_Jy ))
            H_append(Term( mkMat({coor_to_id(i,j,0):3, coor_to_id((i-1)%Lx,(j-1)%Ly,1):3}),coef_Jz ))
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model
def honeycomb_ring(Lx, Ly, **para):
    # assuming PBC
    alpha = para['alpha']
    alpha_X = alpha
    alpha_Y = alpha
    alpha_Z = alpha
    model = Model()
    model.size = Lx*Ly*2
    model.ly = Ly
    model.lx = Lx
    H_append = model.terms.append
    #rand_uni = np.random.uniform
    coor_to_id = lambda x, y, mu: (y*Lx + x)*2 + mu
    for i in range(Lx):
        for j in range(Ly):
            coef_Jx = rnd_beta(alpha_X)*para['jx']
            coef_Jy = rnd_beta(alpha_Y)*para['jy']
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term( mkMat({coor_to_id(i,j,0):1, coor_to_id(i,j,1):1}),coef_Jx ))
            H_append(Term( mkMat({coor_to_id(i,j,0):2, coor_to_id(i,j,1):2}),coef_Jy ))
            H_append(Term( mkMat({coor_to_id(i,j,0):3, coor_to_id(i,j,1):3}),coef_Jz ))
            coef_Jx = rnd_beta(alpha_X)*para['jx']
            coef_Jy = rnd_beta(alpha_Y)*para['jy']
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term( mkMat({coor_to_id(i,j,0):1, coor_to_id(i,(j-1)%Ly,1):1}),coef_Jx ))
            H_append(Term( mkMat({coor_to_id(i,j,0):2, coor_to_id(i,(j-1)%Ly,1):2}),coef_Jy ))
            H_append(Term( mkMat({coor_to_id(i,j,0):3, coor_to_id(i,(j-1)%Ly,1):3}),coef_Jz ))
            coef_Jx = rnd_beta(alpha_X)*para['jx']
            coef_Jy = rnd_beta(alpha_Y)*para['jy']
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term( mkMat({coor_to_id(i,j,0):1, coor_to_id((i-1)%Lx,(j-1)%Ly,1):1}),coef_Jx ))
            H_append(Term( mkMat({coor_to_id(i,j,0):2, coor_to_id((i-1)%Lx,(j-1)%Ly,1):2}),coef_Jy ))
            H_append(Term( mkMat({coor_to_id(i,j,0):3, coor_to_id((i-1)%Lx,(j-1)%Ly,1):3}),coef_Jz ))
            # ring term
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term(mkMat({
                coor_to_id(i,j,0):3, coor_to_id(i,j,1):3, coor_to_id((i+1)%Lx,j,0):3, coor_to_id((i+1)%Lx,j,1):3
                }),coef_Jz))
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term(mkMat({
                coor_to_id(i,j,1):3, coor_to_id((i+1)%Lx, (j+1)%Ly, 0):3, coor_to_id(i,(j-1)%Ly,1):3, coor_to_id((i+1)%Lx,j,0):3
                }),coef_Jz))
            coef_Jz = rnd_beta(alpha_Z)*para['jz']
            H_append(Term(mkMat({
                coor_to_id(i,j,0):3, coor_to_id(i,(j-1)%Ly,1):3, coor_to_id((i+1)%Lx, (j+1)%Ly, 0):3, coor_to_id((i+1)%Lx,j,1):3
                }),coef_Jz))
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model

def clock_transform(ori_model):#x->y,y->z,z->x
    new_model = Model()
    new_model.lx = ori_model.lx
    new_model.ly = ori_model.ly
    new_model.size = ori_model.size
    for k in range(len(ori_model.terms)):
        if len(ori_model.terms[k].mat.Xs^ori_model.terms[k].mat.Zs)==0: #Y interaction
            new_model.terms.append(
            Term(mkMat({i:3 for i in ori_model.terms[k].mat.Zs}),ori_model.terms[k].val)
            )
        elif len(ori_model.terms[k].mat.Xs)==2: # X interaction
            new_model.terms.append(
            Term(mkMat({i:2 for i in ori_model.terms[k].mat.Xs}),ori_model.terms[k].val)
            )
        else: # Z interaction
            new_model.terms.append(
            Term(mkMat({i:1 for i in ori_model.terms[k].mat.Zs}),ori_model.terms[k].val)
            )
    return new_model
def anticlock_transform(ori_model):#z->y, y->x, x->y
    new_model = Model()
    new_model.lx = ori_model.lx
    new_model.ly = ori_model.ly
    new_model.size = ori_model.size
    for k in range(len(ori_model.terms)):
        if len(ori_model.terms[k].mat.Xs^ori_model.terms[k].mat.Zs)==0: #Y interaction
            new_model.terms.append(
            Term(mkMat({i:1 for i in ori_model.terms[k].mat.Zs}),ori_model.terms[k].val)
            )
        elif len(ori_model.terms[k].mat.Xs)==2: # X interaction
            new_model.terms.append(
            Term(mkMat({i:3 for i in ori_model.terms[k].mat.Xs}),ori_model.terms[k].val)
            )
        else: # Z interaction
            new_model.terms.append(
            Term(mkMat({i:2 for i in ori_model.terms[k].mat.Zs}),ori_model.terms[k].val)
            )
    return new_model
def oneDimChain(L, **para):
    # assuming PBC
    alpha = para['alpha']
    alpha_J = alpha
    alpha_h = alpha
    alpha_g = alpha
    model = Model()
    model.size = L
    H_append = model.terms.append
    for i in range(L):
        coef_J = rnd_beta(alpha_J)*para['J']
        coef_h = rnd_beta(alpha_h)*para['h']
        coef_g = rnd_beta(alpha_g)*para['g']
        H_append(Term( mkMat({i:3, (i+1)%L:3}),-coef_J ))
        H_append(Term(mkMat({i:1}),-coef_h))
        H_append(Term(mkMat({(i-1)%L:3, i:1, (i+1)%L:3}),-coef_g))
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model
def oneDimOpenChain(L, **para):
    alpha = para['alpha']
    alpha_J = alpha
    alpha_h = alpha
    alpha_g = alpha
    model = Model()
    model.size = L
    H_append = model.terms.append
    for i in range(L):
        coef_h = rnd_beta(alpha_h)*para['h']
        H_append(Term(mkMat({i:1}),-coef_h))
    for i in range(L-1):
        coef_J = rnd_beta(alpha_J)*para['J']
        H_append(Term( mkMat({i:3, i+1:3}),-coef_J ))
    for i in range(L-2):
        coef_g = rnd_beta(alpha_g)*para['g']
        H_append(Term(mkMat({i:3, i+1:1, i+2:3}),-coef_g))
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model
############################### HYH ###############################
# Use Qutip package to do ED, and transform SBRG to Qutip
paulis=[]
paulis.append(Qobj([[1,0],[0,1]]))
paulis.append(Qobj([[0,1],[1,0]]))
paulis.append(Qobj([[0,-1j], [1j,0]]))
paulis.append(Qobj([[1, 0], [0, -1]]))
def Term_to_Mat(num_qubit, Term):
    '''
    Term is defined in SBRG doc,
    Mat is Qobj defined in Qutip
    '''
    pauli_str = np.zeros(num_qubit)
    list_Xs = list(Term.mat.Xs)
    list_Zs = list(Term.mat.Zs)
    # Find Ys locations
    inter_set = set(list_Xs).intersection(set(list_Zs))
    list_Ys = list(inter_set)
    new_list_Xs = list(set(list_Xs)^inter_set)
    new_list_Zs = list(set(list_Zs)^inter_set)
    pauli_str[new_list_Xs] = np.ones(len(new_list_Xs))
    pauli_str[list_Ys] = np.ones(len(list_Ys))*2
    pauli_str[new_list_Zs] = np.ones(len(new_list_Zs))*3
    pauli_str = list(pauli_str.astype(int))
    #print('pauli str:', pauli_str)
    return tensor([paulis[j] for j in pauli_str])*Term.val
def system_Ham_to_qutip_Ham(system_size, Ham):
    '''
    system.Ham is a list of Terms in SBRG doc
    '''
    system_Ham = Ham.terms
    H = Term_to_Mat(system_size,system_Ham[0])
#     H = Qobj(np.zeros((2**system_size, 2**system_size)).tolist)
    for term_id in range(1,len(system_Ham)):
        H += Term_to_Mat(system_size,system_Ham[term_id])
    return H
def system_RCC_to_qutip_RCC(system_size, system_RCC):
    '''
    system_RCC is a list of Terms in SBRG doc
    Urcc = R1R2R3...
    '''
    RCC = ((1j*np.pi/4)*Term_to_Mat(system_size, system_RCC[0])).expm()
    for i in range(1, len(system_RCC)):
        RCC = RCC*((1j*np.pi/4)*Term_to_Mat(system_size, system_RCC[i])).expm()
    return RCC
def get_full_spectrum(SBRG_res):
    '''
    SBRG_res is an object of SBRG class
    '''
    list_basis = list(product([-1,1],repeat = SBRG_res.size))
    spectrum = []
    for basis_id in range(0, len(list_basis)):
        energy = 0
        for Heff_id in range(0, len(SBRG_res.Heff)):
            list_of_taus = list(SBRG_res.Heff[Heff_id].mat.Zs)
            energy += np.prod(np.asarray(list_basis[basis_id])[np.asarray(list_of_taus)])*SBRG_res.Heff[Heff_id].val
        spectrum.append(energy)
    return spectrum
############################### G-SPT models ###############################
def oneDimChain(L, **para):
    # assuming PBC
    alpha = para['alpha']
    alpha_J = alpha
    alpha_h = alpha
    alpha_g = alpha
    model = Model()
    model.size = L
    H_append = model.terms.append
    for i in range(L):
        coef_J = rnd_beta(alpha_J)*para['J']
        coef_h = rnd_beta(alpha_h)*para['h']
        coef_g = rnd_beta(alpha_g)*para['g']
        H_append(Term( mkMat({i:3, (i+1)%L:3}),-coef_J ))
        H_append(Term(mkMat({i:1}),-coef_h))
        H_append(Term(mkMat({(i-1)%L:3, i:1, (i+1)%L:3}),-coef_g))
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model
def oneDimOpenChain(L, **para):
    alpha = para['alpha']
    alpha_J = alpha
    alpha_h = alpha
    alpha_g = alpha
    model = Model()
    model.size = L
    H_append = model.terms.append
    for i in range(L):
        coef_h = rnd_beta(alpha_h)*para['h']
        H_append(Term(mkMat({i:3}),-coef_h))
#         H_append(Term(mkMat({i:3}),-para['h']))
    for i in range(L-1):
        coef_J = rnd_beta(alpha_J)*para['J']
        H_append(Term( mkMat({i:3, i+1:3}),-coef_J ))
#         H_append(Term( mkMat({i:3, i+1:3}),-para['J'] ))
    for i in range(L-2):
        coef_g = rnd_beta(alpha_g)*para['g']
        H_append(Term(mkMat({i:3, i+1:1, i+2:3}),-coef_g))
#         H_append(Term(mkMat({i:3, i+1:1, i+2:3}),-para['g']))
#     for i in range(int((L/2)-2)):
#         coef_V = rnd_beta(alpha_g)*para['V']
#         H_append(Term(mkMat({i:1, i+1:3, i+2:3}),-coef_V))
#         coef_V = rnd_beta(alpha_g)*para['V']
#         H_append(Term(mkMat({L-1-i:1, L-1-(i+1):3, L-1-(i+2):3}),-coef_V))
    coef_V = rnd_beta(alpha_g)*para['V']
    H_append(Term(mkMat({L-1:1, L-2:3, L-3:3}),-para['V']))
    coef_V = rnd_beta(alpha_g)*para['V']
    H_append(Term(mkMat({0:1, 1:3, 2:3}),-para['V']))
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model
def Ising(L, **para):
    # L - number of sites (assuming PBC)
    # model - a dict of model parameters
    try: # set parameter alpha
        alpha = para['alpha']
        alpha_J = alpha
        alpha_h = alpha
    except:
        alpha_J = para.get('alpha_J',1)
        alpha_h = para.get('alpha_h',1)
    model = Model()
    model.size = L
    # translate over the lattice by deque rotation
    H_append = model.terms.append
    tmp_rnd_beta = random.betavariate
    for i in range(L):
        H_append(Term(mkMat({i: 3, (i+1)%L: 3}), para['J']*tmp_rnd_beta(alpha_J, 1)))
        H_append(Term(mkMat({i: 1}), para['h']*tmp_rnd_beta(alpha_h, 1)))
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model
def OpenIsing(L,**para):
    try: # set parameter alpha
        alpha = para['alpha']
        alpha_J = alpha
        alpha_h = alpha
    except:
        alpha_J = para.get('alpha_J',1)
        alpha_h = para.get('alpha_h',1)
    model = Model()
    model.size = L
    # translate over the lattice by deque rotation
    H_append = model.terms.append
    tmp_rnd_beta = random.betavariate
    
#     for i in range(L-1):
#         H_append(Term(mkMat({i: 3, (i+1): 3}),-para['J']))
#     for i in range(L):
#         H_append(Term(mkMat({i: 1}), -para['g']))
#         H_append(Term(mkMat({i: 3}), -para['h']))
    
    for i in range(L-1):
        H_append(Term(mkMat({i: 3, (i+1): 3}), -para['J']*tmp_rnd_beta(alpha_J, 1)))
    for i in range(L):
        H_append(Term(mkMat({i: 1}), -para['g']*tmp_rnd_beta(alpha_h, 1)))
        H_append(Term(mkMat({i: 3}), -para['h']*tmp_rnd_beta(alpha_h, 1)))
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model
############################### HYH ###############################
def save_model(filename,model):
    data = {}
    data['size']=model.size
    data['numberofterms']=len(model.terms)
    for i in range(len(model.terms)):
        data['{}val'.format(i)]=model.terms[i].val
        data['{}Xs'.format(i)] = list(model.terms[i].mat.Xs)
        data['{}Zs'.format(i)] = list(model.terms[i].mat.Zs)
    with open(filename+'.json','wb') as writefile:
        pickle.dump(data,writefile)
def load_model(filename):
    with open('./'+filename+'.json','rb') as readfile:
        read_data = pickle.load(readfile)
    read_model = Model()
    read_model.size = read_data['size']
    H_append = read_model.terms.append
    for i in range(read_data['numberofterms']):
        H_append(Term(mkMat(set(read_data['{}Xs'.format(i)]),\
                            set(read_data['{}Zs'.format(i)])),read_data['{}val'.format(i)]))
    return read_model
############################### HYH ###############################
# def Term_to_Mat(num_qubit, Term):
#     '''
#     Term is defined in SBRG doc,
#     Mat is Qobj defined in Qutip
#     '''
#     pauli_str = np.zeros(num_qubit)
#     list_Xs = list(Term.mat.Xs)
#     list_Zs = list(Term.mat.Zs)
#     # Find Ys locations
#     inter_set = set(list_Xs).intersection(set(list_Zs))
#     list_Ys = list(inter_set)
#     new_list_Xs = list(set(list_Xs)^inter_set)
#     new_list_Zs = list(set(list_Zs)^inter_set)
#     pauli_str[new_list_Xs] = np.ones(len(new_list_Xs))
#     pauli_str[list_Ys] = np.ones(len(list_Ys))*2
#     pauli_str[new_list_Zs] = np.ones(len(new_list_Zs))*3
#     pauli_str = list(pauli_str.astype(int))
#     #print('pauli str:', pauli_str)
#     return tensor([paulis[j] for j in pauli_str])*Term.val
# def system_Ham_to_qutip_Ham(system_size, system_Ham):
#     '''
#     system.Ham is a list of Terms in SBRG doc
#     '''
#     H = Qobj(np.zeros((2**system_size, 2**system_size)).tolist)
#     for term_id in range(len(system_Ham)):
# #         print('process: {}%'.format((term_id+1)/len(system_Ham)*100))
#         #clear_output(wait=True)
#         H += Term_to_Mat(system_size,system_Ham[term_id])
#     return H
# def system_RCC_to_qutip_RCC(system_size, system_RCC):
#     '''
#     system_RCC is a list of Terms in SBRG doc
#     Urcc = R1R2R3...
#     '''
#     RCC = ((1j*np.pi/4)*Term_to_Mat(system_size, system_RCC[0])).expm()
#     for i in range(1, len(system_RCC)):
#         RCC = RCC*((1j*np.pi/4)*Term_to_Mat(system_size, system_RCC[i])).expm()
#     return RCC
# def get_full_spectrum(SBRG_res):
#     '''
#     SBRG_res is an object of SBRG class
#     '''
#     list_basis = list(product([-1,1],repeat = SBRG_res.size))
#     spectrum = []
#     for basis_id in range(0, len(list_basis)):
#         energy = 0
#         for Heff_id in range(0, len(SBRG_res.Heff)):
#             list_of_taus = list(SBRG_res.Heff[Heff_id].mat.Zs)
#             energy += np.prod(np.asarray(list_basis[basis_id])[np.asarray(list_of_taus)])*SBRG_res.Heff[Heff_id].val
#         spectrum.append(energy)
#     return spectrum





############################### HYH ###############################


# Toolbox 
# I/O 
# JSON pickle: export to communicate with Mathematica 
import jsonpickle
def export(filename, obj):
    with open(filename + '.json', 'w') as outfile:
        outfile.write(jsonpickle.encode(obj))
def export_Ham(filename, ham):
    export(filename, [[term.val,[list(term.mat.Xs),list(term.mat.Zs)]] for term in ham])
import pickle
# pickle: binary dump and load for python.
def dump(filename, obj):
    with open(filename + '.dat', 'bw') as outfile:
        pickle.dump(obj, outfile)
def load(filename):
    with open(filename + '.dat', 'br') as infile:
        return pickle.load(infile)

