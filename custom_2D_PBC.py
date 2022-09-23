import SBRG.py
import matplotlib.pyplot as plt
import time

#Need to create an index mapping to convert a 2D Pauli array into a 1D Pauli array
#For example, I have the term  A s_(0,0) s_(1,0) + B s_(0,0) s_(0,1) + C s_(0,0)s_(-1, 0) + D s_(0,0)s_(0, -1)
#Write as A s_01 + B s_02 + C s_03 + D s_04

def index_map(x, y, Lx, Ly): #Takes as inout the x,y index of the lattice point in a Lx x Ly size square lattice
    ind = y + Ly*x
    if ind>Lx*Ly - 1:
        print("Out of Bounds")
        return None
    else: return ind



#Random 2D Ising model with x and y length given by Lx, Ly. Hamiltonian terms include ferromagnetic J, paramagnetic h and
#potential term mu

def custom_2D_Ising(Lx, Ly, **para):
    # model - a dict of model parameters
    try: # set parameter alpha
        alpha = para['alpha']
        alpha_J = alpha
        alpha_mu = alpha
        alpha_h = alpha
    except:
        alpha_J = para.get('alpha_J',1)
        alpha_mu = para.get('alpha_mu',1)
        alpha_h = para.get('alpha_h',1)
    model = Model()
    model.size = Lx*Ly
    H_append = model.terms.append
    rnd_beta = random.betavariate
    for x in range(Lx):
        for y in range(Ly):

            ind_0 = index_map(x, y, Lx, Ly)
            ind_x_nbh = index_map((x+1)%Lx, y, Lx, Ly)
            ind_y_nbh = index_map(x, (y+1)%Ly, Lx, Ly)
            
            #Ferromagnetic term
            H_append(Term(mkMat({ind_0: 3, ind_x_nbh: 3}), -para['J']*rnd_beta(alpha_J, 1)))
            H_append(Term(mkMat({ind_0: 3, ind_y_nbh: 3}), -para['J']*rnd_beta(alpha_J, 1)))
            
            #Potential Term
            H_append(Term(mkMat({ind_0: 3}), -para['mu']*rnd_beta(alpha_mu, 1)))
            
            #Paramagnetic term
            H_append(Term(mkMat({ind_0: 1}), -para['h']*rnd_beta(alpha_h, 1)))
    model.terms = [term for term in model.terms if abs(term.val) > 0]
    return model


#Two-point correlator <Z_(x1, y1) Z_(x2, y2)>
def spin_spin_op(x1, y1, x2, y2, Lx, Ly):
    ind_i = index_map(x1, y1, Lx, Ly)
    ind_j = index_map(x2, y2, Lx, Ly)
    return Ham([Term(mkMat({ind_i: 3, ind_j: 3}), 1)])
    #return [Term(mkMat({ind_i: 3, ind_j: 3}), 1)]



