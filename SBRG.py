# dict of single-bit dot product rules
DOT_RULES = {(0,0): (0,0),
             (0,1): (1,0),
             (0,2): (2,0),
             (0,3): (3,0),
             (1,0): (1,0),
             (2,0): (2,0),
             (3,0): (3,0),
             (1,1): (0,0),
             (2,2): (0,0),
             (3,3): (0,0),
             (1,2): (3,1),
             (2,3): (1,1),
             (3,1): (2,1),
             (3,2): (1,-1),
             (2,1): (3,-1),
             (1,3): (2,-1)}
# merge dicts of two matrices
def merge(mat1_get, mat2_get, mat1_keys, mat2_keys):
    return [[i, mat1_get(i,0), mat2_get(i,0)] for i in mat1_keys | mat2_keys]
# carry out DOT_RULES over index pairs
def dotover(merged):
    for triple in merged:
        triple[1], triple[2] = DOT_RULES[(triple[1], triple[2])]
# get overall exponent
def powofi(merged, n0 = 0):
    return (sum(triple[2] for triple in merged) + n0)%4
# collect new mat
def newmat(merged):
    return {i: mu for [i, mu, n] in merged if mu != 0}
# dot product
def dot(term1, term2, n0 = 0):
    merged = merge(term1[0].get, term2[0].get, term1[0].keys(), term2[0].keys())
    dotover(merged)
    return [newmat(merged), term1[1]*term2[1]*(-1)**(powofi(merged, n0)/2)]
# find rotation to diagonalize mat0 to the current bit site i_now
def find_rotation(mat0, i_now):
    # i_12 - site of first 1 or 2, i_3 - site of last 3
    i_12 = -1 # initialize to -1 (unphysical)
    i_3 = -1 # initialize to -1 (unphysical)
    count_3 = 0 # count the num of 3
    for (i, mu) in mat0.items():
        if mu == 3: # if find 3
            count_3 += 1
            if i >= i_now: # if after i_now
                i_3 = i # update i_3
        elif i >= i_now and (mu == 1 or mu == 2): # if find first 1 or 2 after i_now
            i_12 = i # set i_12 and break
            break
    Us = []
    if i_12 >= 0: # if 1 or 2 exist, use it as pivot of rotation
        # require a C4 rotation
        Us.append(['C4', dot([{i_12: 3},1], [mat0,1], 1)])
        if i_12 != i_now: # if i_12 not at the required site
            Us.append(['SWAP',[i_now, i_12]]) # additional SWAP required
    elif i_3 >= 0: # if no 1 or 2, but 3 exist
        if count_3 > 1: # if there are more than one 3
            # require double C4
            if mat0.get(i_now,0) == 3: # if i_now sits on a site of 3
                i_3 = i_now # switch i_3 to that, so as to avoid additional SWAP
            Us.append(['C4', dot([{i_3: 1},1], [mat0,1], 1)])
            Us.append(['C4', [{i_3: 2}, -1]])       
        if i_3 != i_now: # if i_3 not at the reqired site
            Us.append(['SWAP',[i_now, i_3]]) # additional SWAP required 
    return Us
# single unitary transform
# C4 transformation
def C4(mat_C4, val_C4, H):
    mat_C4_get = mat_C4.get
    mat_C4_keys = mat_C4.keys()
    # if non-commuting, do the dot product, otherwise do nothing
    for term in H:
        mat_keys = term[0].keys()
        if mat_C4_keys & mat_keys: # if keys intersects
            # merge keys
            mat_get = term[0].get
            merged = merge(mat_get, mat_C4_get, mat_keys, mat_C4_keys)
            if sum(1 for [i, mu1, mu2] in merged
                   if mu1 != 0 and mu2 != 0 and mu1 != mu2)%2:
                # if not commute, perform C4
                dotover(merged)
                term[0] = newmat(merged)
                term[1] *= val_C4*(-1)**(powofi(merged, 1)/2) 
# SWAP gate
def swap(i0, i1, H):
    for term in H:
        mat_keys = term[0].keys()
        if i0 in mat_keys: # i0 intersects
            if i1 in mat_keys: # both i0, i1 intersect
                term[0][i0], term[0][i1] = term[0][i1], term[0][i0]
            else: # only i0 intersects
                term[0][i1] = term[0].pop(i0) # update i0 -> i1
        else: # i0 not there
            if i1 in mat_keys: # only i1 intersects
                term[0][i0] = term[0].pop(i1) # update i1->i0
# perform unitary transforms Us in series to H
def unitary_fd(Us, H):
    for [gate, para] in Us:
        if gate == 'C4': # C4 gate, para = C4 generator
            C4(para[0], para[1], H)   
        elif gate == 'SWAP': # SWAP gate, para = SWAP positions
            swap(para[0], para[1], H)
def unitary_bk(Us, H):
    for [gate, para] in reversed(Us): # access Us in reversed order
        if gate == 'C4': # C4 gate, para = C4 generator
            C4(para[0], -para[1], H) # C4 rotation in opposite direction
        elif gate == 'SWAP': # SWAP gate, para = SWAP positions
            swap(para[0], para[1], H)
# key functions for sorting
def term_mat(term):
    return tuple(term[0].items())
def term_val(term):
    return abs(term[1])
# cal perturbation Hamiltonian
from itertools import combinations
def perturbation(H_offdiag, h0, i_now, min_scale, max_rate, trash_add):
    # preselect pair of terms above min_scale
    min_prod = abs(h0*min_scale)
    H_prod = [[merge(term1[0].get, term2[0].get, term1[0].keys(), term2[0].keys()),
               term1[1]*term2[1]/h0] 
              for (term1, term2) in combinations(H_offdiag, 2)
              if abs(term1[1]*term2[1]) > min_prod]
    # quench terms from anticommuting products
    for term in H_prod:
        if sum(1 for [i, mu1, mu2] in term[0]
               if mu1 != 0 and mu2 != 0 and mu1 != mu2)%2: # if anticommute
            term[1] = 0 # quench the coefficient
    # sort by val
    H_prod = [[merged, val] for [merged, val] 
              in H_prod if abs(val) > min_scale]
    H_prod.sort(key=term_val)
    # term number truncation
    max_len = round(max_rate*len(H_offdiag))
    if len(H_prod) > max_len:
        trash_add([val for [merged, val] in H_prod[:-max_len]])
        H_prod = H_prod[-max_len:]
    # carryout the matrix product
    for term in H_prod:
        dotover(term[0])
        term[1] *= (-1)**(powofi(term[0])/2)
        term[0] = newmat(term[0])
        # apply H0 dot product
        mu_now = term[0].get(i_now,0)
        if mu_now == 0:
            term[0][i_now] = 3
        else: # mu_now = 3
            del term[0][i_now]
    # the backward correction to the leading energy scale
    H_prod.append([{i_now: 3}, sum(val**2 for [mat, val] in H_offdiag)/(2*h0)])
    return H_prod
# judge if a matrix is identity starting from i_now position
def is_iden(mat, i_now):
    return max(mat.keys()) <= i_now
# SBRG class
from copy import deepcopy
from itertools import chain
class SBRG:
    def __init__(self, model):
        self.tol = 1.e-8
        self.max_rate = 2.
        self.recover = True
        self.N = model['bits'] # total number of bits
        self.Neff = 0 # number of effective bits that has been discovered
        self.H = deepcopy(model['H']) # physical Hamiltonian
        self.Heff = [] # effective Hamiltonian
        self.gates = [] # Clifford gates collected along the way
        self.trash = [] # terms truncated in 2nd order perturbation
    # one RG step forward
    def next_step(self):
        i_now = self.Neff
        H_UV = self.H
        if len(H_UV) == 0: # return if H = 0
            # some emergent qubits are zero modes
            self.Neff = self.N # no physical qubits left
            return H_UV
        # find the leading energy scale
        [mat_H0, h0] = max(H_UV, key=term_val)
        min_scale = abs(h0*self.tol)
         # diagonalize the leading term
        Us = find_rotation(mat_H0, i_now)
        self.gates.append(Us)
        # perform unitary transforms to the whole Hamiltonian
        unitary_fd(Us, H_UV)
        # and mark out diag and offdiag terms
        H_gather = [(0 < term[0].get(i_now,0) < 3, term) for term in H_UV]
        # off diagonal terms goes to H_offdiag
        H_offdiag = [term for (is_offdiag, term) in H_gather 
                     if is_offdiag]
        # H_prod = H_offdiag^2/(2*h0) -> 2nd order perturbation
        H_prod = perturbation(H_offdiag, h0, i_now,
                              min_scale, self.max_rate, self.trash.extend)
        # add the 2nd order perturbation with the diag part of H
        H_prod += [term for (is_offdiag, term) in H_gather 
                   if not is_offdiag]
        # prepare to merge similar terms
        H_prod.sort(key=term_mat) # strategy: merge by sorting
        mat_last = {}
        H_IR = []
        for [mat, val] in H_prod: # loop of merging
            if abs(val) > min_scale:
                if mat != mat_last: # if mat is new
                    H_IR.append([mat, val]) # add to H
                    mat_last = mat # update mat_last
                else: # if mat == mat_last
                    H_IR[-1][1] += val # add val to
        # mark out identity terms in the remaining physical space
        H_gather = [(is_iden(term[0], i_now), term) for term in H_IR]
        self.H = [term for (iden, term) in H_gather if not iden]
        self.Heff.extend([term for (iden, term) in H_gather if iden])
        self.Neff = i_now + 1 # tau bits counter inc
        return self.H
    # RG flow
    def flow(self, step = float('inf')):
        # carry out RG flow
        if step > (self.N - self.Neff):
            step = self.N - self.Neff
        stp_count = 0
        while (self.Neff < self.N and stp_count < step):
            self.next_step()
            stp_count += 1
        if self.recover: # recover original locality
            blk = list(range(self.N)) # to keep track of the action of SWAP gates
            for l, Us in enumerate(reversed(self.gates)):
                if len(Us)>0 and Us[-1][0] == 'SWAP': # if Us ends with a SWAP gate
                    i, j = Us[-1][1]
                    # perform swap to the C4 gates in IR direction
                    swap(i, j, chain(*((C4[1] for C4 in C4s) for C4s in self.gates[-l:])))
                    blk[i], blk[j] = blk[j], blk[i]
                    del Us[-1] # drop the SWAP gate
            imap = {fr: to for (to, fr) in enumerate(blk)}
            for term in self.Heff: # update Heff mat indices
                term[0] = {imap[i]: mu for (i, mu) in term[0].items()}
# Model Hamiltonians
import random
# H of TFIsing
def TFIsing(L, **para):
    # L - number of sites (assuming PBC)
    # model - a dict of model parameters {J, alpha_J, K, alpha_K, h, alpha_h}
    H = []
    # translate over the lattice by deque rotation
    H_append = H.append
    rnd_beta = random.betavariate
    for i in range(L):
        H_append([{i: 1, (i+1)%L: 1}, para['J']*rnd_beta(para['alpha_J'], 1)])
        H_append([{i: 3, (i+1)%L: 3}, para['K']*rnd_beta(para['alpha_K'], 1)])
        H_append([{i: 3}, para['h']*rnd_beta(para['alpha_h'], 1)])
    H = [term for term in H if abs(term[1]) > 0]
    return {'bits': L, 'H': H}
# Toolbox 
# I/O 
# JSON pickle: export to communicate with Mathematica 
import jsonpickle
def export(filename, obj):
    with open(filename + '.json', 'w') as outfile:
        outfile.write(jsonpickle.encode(obj))
import pickle
# pickle: binary dump and load for python.
def dump(filename, obj):
    with open(filename + '.dat', 'bw') as outfile:
        pickle.dump(obj, outfile)
def load(filename):
    with open(filename + '.dat', 'br') as infile:
        return pickle.load(infile)
# visualization
import matplotlib.pyplot as plt
def hist_plot(data):
    if len(data) > 1:
        plt.hist(data, 30)
        plt.xlabel("Energy Scale")
        plt.ylabel("Frequency")
        plt.show()
    else:
        print('hist_plot: input has only one data point.')