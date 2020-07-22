# SBRG(2.0): Spectrum Bifurcation Renormalization Group
(This is an update of the SBRG algorithm. New version contains link to `Qutip` packages, calculating physical observables, and etc)

SBRG is an algorithm motivated by RSRG-X. Instead of getting just ground state information, SBRG will calculate the effective Hamiltonian for many-body localized(MBL) systems. You can use the effective Hamiltonian to calculate the full spectrum of the original many-body system. This algorithm has been applied to study:
- MBL physics
- Holography network
- Infinite-randomness critical point(IRCP) in 1D and 2D
- IRCP with symmetry protected topological states

You need Numpy, Numba, Qutip (minimal) to run the SBRG(2.0). If you do not want to bother install all packages, we have create an enviroment file to install all the things you need once for all. To install the environment file, you need to type the following command into your terminal (assume you have anaconda installed):

`conda env create -f SBRG_env.yml`
