import math
import numpy as np

def cutoff_function(distance, cutoff_radius):
    '''
    cutoff_function returns the value of the cutoff function as defined in:

    "Atom Centered Symmetry Fucntions for Contructing High-Dimentional
    Neural Network Potentials",
    by Jorg Behler, J. Chem. Phys. 134, 074106 (2011)
    '''
    if distance > cutoff_radius:
        return 0.0
    return 0.5 * (math.cos((math.pi*distance)/cutoff_radius) + 1.0)

def symmetry_function_1(cutoff, mdm):
    '''
    This function generates the first symmetry function G1 as defined in:

    "Atom Centered Symmetry Fucntions for Contructing High-Dimentional
    Neural Network Potentials",
    by Jorg Behler, J. Chem. Phys. 134, 074106 (2011)
    '''
    num_atoms = len(mdm)
    sym_func_1 = np.zeros(num_atoms)
    for i, distance_set in enumerate(mdm):
        for distance in distance_set:
            sym_func_1[i] += cutoff_function(distance, cutoff)

    return sym_func_1

def symmetry_function_2(cutoff, rs, eta, mdm):
    '''
    This function generates the second symmetry function G2 as defined in:

    "Atom Centered Symmetry Fucntions for Contructing High-Dimentional
    Neural Network Potentials",
    by Jorg Behler, J. Chem. Phys. 134, 074106 (2011)
    '''
    num_atoms = len(mdm)
    sym_func_2 = np.zeros(num_atoms)
    for i, distance_set in enumerate(mdm):
        for distance in distance_set:
            val = cutoff_function(distance, cutoff)
            if val != 0.0:
                sym_func_2[i] += (math.exp(-eta*((distance-rs)**2)) * val)

    return sym_func_2

def symmetry_function_3(cutoff, kappa, mdm):
    '''
    This function generates the third symmetry function G3 as defined in:

    "Atom Centered Symmetry Fucntions for Contructing High-Dimentional
    Neural Network Potentials",
    by Jorg Behler, J. Chem. Phys. 134, 074106 (2011)
    '''
    num_atoms = len(mdm)
    sym_func_3 = np.zeros(num_atoms)
    for i, distance_set in enumerate(mdm):
        for distance in distance_set:
            val = cutoff_function(distance, cutoff)
            if val != 0.0:
                sym_func_3[i] += (math.cos(kappa * distance) * val)

    return sym_func_3

def symmetry_function_4(cutoff, lamb, zeta, eta, mdm, mdv):
    '''
    This function generates the forth symmetry function G4 as defined in:

    "Atom Centered Symmetry Fucntions for Contructing High-Dimentional
    Neural Network Potentials",
    by Jorg Behler, J. Chem. Phys. 134, 074106 (2011)
    '''
    num_atoms = len(mdm)
    sym_func_4 = np.zeros(num_atoms)

    # vectors from atom i to j, and from i to k.
    Rij = np.zeros(3)
    Rik = np.zeros(3)

    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                continue
            FcRij = cutoff_function(mdm[i][j], cutoff)
            if FcRij == 0.0:
                continue
            Rij = mdv[i][j][:]
            for k in range(num_atoms):
                if i == k or j == k:
                    continue
                FcRik = cutoff_function(mdm[i][k], cutoff)
                if FcRik == 0.0:
                    continue
                FcRjk = cutoff_function(mdm[j][k], cutoff)
                if FcRjk == 0.0:
                    continue
                Rik = mdv[i][k][:]
                # the cos of the angle is the dot product devided
                # by the multiplication of the magnitudes. the
                # magnitudes are the distances in the mdm.
                cos_theta_ijk = np.dot(Rij, Rik) / (mdm[i][j] * mdm[i][k])
                sym_func_4[i] += (((1.0+lamb*cos_theta_ijk)**zeta) *
                                  (math.exp(-eta*(mdm[i][j]**2 + mdm[i][k]**2 + mdm[j][k]**2))) *
                                  FcRij * FcRik * FcRjk)

        sym_func_4[i] *= (2**(1.0 - zeta))

    return sym_func_4

def symmetry_function_5(cutoff, lamb, zeta, eta, mdm=None, mdv=None):
    '''
    This function generates the fifth symmetry function G5 as defined in:

    "Atom Centered Symmetry Fucntions for Contructing High-Dimentional
    Neural Network Potentials",
    by Jorg Behler, J. Chem. Phys. 134, 074106 (2011)
    '''
    num_atoms = len(mdm)
    sym_func_5 = np.zeros(num_atoms)

    # vectors from atom i to j, and from i to k.
    Rij = np.zeros(3)
    Rik = np.zeros(3)

    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                continue
            FcRij = cutoff_function(mdm[i][j], cutoff)
            if FcRij == 0.0:
                continue
            Rij = mdv[i][j]
            for k in range(num_atoms):
                if i == k:
                    continue
                FcRik = cutoff_function(mdm[i][k], cutoff)
                if FcRik == 0.0:
                    continue
                Rik = mdv[i][k]
                # the cos of the angle is the dot product devided
                # by the multiplication of the magnitudes. the
                # magnitudes are the distances in the mdm.
                cos_theta_ijk = np.dot(Rij, Rik) / (mdm[i][j] * mdm[i][k])
                sym_func_5[i] += (((1.0+lamb*cos_theta_ijk)**zeta) *
                                  (math.exp(-eta*(mdm[i][j]**2 + mdm[i][k]**2))) *
                                  FcRij * FcRik)

        sym_func_5[i] *= (2**(1.0 - zeta))

    return sym_func_5

def symmetry_function_set(structure):
    '''
    This function returns a set of 79 symmetry functions that describe
    the local atomic environment for each atom in a structure.
    The set was chosen after some experimentation. So while the set is
    good, it might not be the best, or the most efficient way to
    describe the local env. of atoms.
    '''
    all_sym_functions = np.zeros(shape=(structure.num_atoms, 79))
    mdm = structure.min_dist_matrix()
    mdv = structure.min_dist_vector(mdm=mdm)
    print("Working on the symmetry function 1 set...")
    all_sym_functions[:, 0] = symmetry_function_1(1.0, mdm)
    all_sym_functions[:, 1] = symmetry_function_1(1.2, mdm)
    all_sym_functions[:, 2] = symmetry_function_1(1.4, mdm)
    all_sym_functions[:, 3] = symmetry_function_1(1.6, mdm)
    all_sym_functions[:, 4] = symmetry_function_1(1.8, mdm)
    all_sym_functions[:, 5] = symmetry_function_1(2.0, mdm)
    all_sym_functions[:, 6] = symmetry_function_1(2.2, mdm)
    all_sym_functions[:, 7] = symmetry_function_1(2.4, mdm)
    all_sym_functions[:, 8] = symmetry_function_1(2.6, mdm)
    all_sym_functions[:, 9] = symmetry_function_1(2.8, mdm)
    all_sym_functions[:, 10] = symmetry_function_1(3.0, mdm)
    all_sym_functions[:, 11] = symmetry_function_1(3.2, mdm)
    all_sym_functions[:, 12] = symmetry_function_1(3.4, mdm)
    all_sym_functions[:, 13] = symmetry_function_1(3.6, mdm)
    all_sym_functions[:, 14] = symmetry_function_1(3.8, mdm)
    all_sym_functions[:, 15] = symmetry_function_1(4.0, mdm)
    all_sym_functions[:, 16] = symmetry_function_1(4.2, mdm)
    all_sym_functions[:, 17] = symmetry_function_1(4.4, mdm)
    all_sym_functions[:, 18] = symmetry_function_1(4.6, mdm)
    all_sym_functions[:, 19] = symmetry_function_1(4.8, mdm)
    all_sym_functions[:, 20] = symmetry_function_1(5.0, mdm)
    all_sym_functions[:, 21] = symmetry_function_1(5.2, mdm)
    all_sym_functions[:, 22] = symmetry_function_1(5.4, mdm)
    all_sym_functions[:, 23] = symmetry_function_1(5.6, mdm)
    all_sym_functions[:, 24] = symmetry_function_1(5.8, mdm)
    all_sym_functions[:, 25] = symmetry_function_1(6.0, mdm)

    print("Working on the symmetry function 2 set...")
    all_sym_functions[:, 26] = symmetry_function_2(5.0, 2.0, 12.0, mdm)
    all_sym_functions[:, 27] = symmetry_function_2(5.0, 2.2, 12.0, mdm)
    all_sym_functions[:, 28] = symmetry_function_2(5.0, 2.4, 12.0, mdm)
    all_sym_functions[:, 29] = symmetry_function_2(5.0, 2.6, 12.0, mdm)
    all_sym_functions[:, 30] = symmetry_function_2(5.0, 2.8, 12.0, mdm)
    all_sym_functions[:, 31] = symmetry_function_2(5.0, 3.0, 12.0, mdm)
    all_sym_functions[:, 32] = symmetry_function_2(5.0, 3.2, 12.0, mdm)
    all_sym_functions[:, 33] = symmetry_function_2(5.0, 3.4, 12.0, mdm)
    all_sym_functions[:, 34] = symmetry_function_2(5.0, 3.6, 12.0, mdm)
    all_sym_functions[:, 35] = symmetry_function_2(5.0, 3.8, 12.0, mdm)
    all_sym_functions[:, 36] = symmetry_function_2(5.0, 4.0, 12.0, mdm)
    all_sym_functions[:, 37] = symmetry_function_2(5.0, 4.2, 12.0, mdm)
    all_sym_functions[:, 38] = symmetry_function_2(5.0, 4.4, 12.0, mdm)
    all_sym_functions[:, 39] = symmetry_function_2(5.0, 4.6, 12.0, mdm)
    all_sym_functions[:, 40] = symmetry_function_2(5.0, 4.8, 12.0, mdm)

    print("Working on the symmetry function 3 set...")
    all_sym_functions[:, 41] = symmetry_function_3(5.0, 1.0, mdm)
    all_sym_functions[:, 42] = symmetry_function_3(5.0, 1.5, mdm)
    all_sym_functions[:, 43] = symmetry_function_3(5.0, 2.0, mdm)
    all_sym_functions[:, 44] = symmetry_function_3(5.0, 2.5, mdm)
    all_sym_functions[:, 45] = symmetry_function_3(5.0, 3.0, mdm)
    all_sym_functions[:, 46] = symmetry_function_3(5.0, 3.5, mdm)
    all_sym_functions[:, 47] = symmetry_function_3(5.0, 4.0, mdm)
    all_sym_functions[:, 48] = symmetry_function_3(5.0, 4.5, mdm)

    print("Working on the symmetry function 4 set...")
    all_sym_functions[:, 49] = symmetry_function_4(5.0, 1.0, 1.0, 0.05, mdm, mdv)
    all_sym_functions[:, 50] = symmetry_function_4(5.0, 1.0, 2.0, 0.05, mdm, mdv)
    all_sym_functions[:, 51] = symmetry_function_4(5.0, 1.0, 3.0, 0.05, mdm, mdv)
    all_sym_functions[:, 52] = symmetry_function_4(5.0, 1.0, 4.0, 0.05, mdm, mdv)
    all_sym_functions[:, 53] = symmetry_function_4(5.0, 1.0, 5.0, 0.05, mdm, mdv)
    all_sym_functions[:, 54] = symmetry_function_4(5.0, 1.0, 1.0, 0.07, mdm, mdv)
    all_sym_functions[:, 55] = symmetry_function_4(5.0, 1.0, 2.0, 0.07, mdm, mdv)
    all_sym_functions[:, 56] = symmetry_function_4(5.0, 1.0, 3.0, 0.07, mdm, mdv)
    all_sym_functions[:, 57] = symmetry_function_4(5.0, 1.0, 4.0, 0.07, mdm, mdv)
    all_sym_functions[:, 58] = symmetry_function_4(5.0, 1.0, 5.0, 0.07, mdm, mdv)
    all_sym_functions[:, 59] = symmetry_function_4(5.0, 1.0, 1.0, 0.09, mdm, mdv)
    all_sym_functions[:, 60] = symmetry_function_4(5.0, 1.0, 2.0, 0.09, mdm, mdv)
    all_sym_functions[:, 61] = symmetry_function_4(5.0, 1.0, 3.0, 0.09, mdm, mdv)
    all_sym_functions[:, 62] = symmetry_function_4(5.0, 1.0, 4.0, 0.09, mdm, mdv)
    all_sym_functions[:, 63] = symmetry_function_4(5.0, 1.0, 5.0, 0.09, mdm, mdv)

    print("Working on the symmetry function 5 set...")
    all_sym_functions[:, 64] = symmetry_function_5(5.0, 1.0, 1.0, 0.3, mdm, mdv)
    all_sym_functions[:, 65] = symmetry_function_5(5.0, 1.0, 2.0, 0.3, mdm, mdv)
    all_sym_functions[:, 66] = symmetry_function_5(5.0, 1.0, 3.0, 0.3, mdm, mdv)
    all_sym_functions[:, 67] = symmetry_function_5(5.0, 1.0, 4.0, 0.3, mdm, mdv)
    all_sym_functions[:, 68] = symmetry_function_5(5.0, 1.0, 5.0, 0.3, mdm, mdv)
    all_sym_functions[:, 69] = symmetry_function_5(5.0, 1.0, 1.0, 0.4, mdm, mdv)
    all_sym_functions[:, 70] = symmetry_function_5(5.0, 1.0, 2.0, 0.4, mdm, mdv)
    all_sym_functions[:, 71] = symmetry_function_5(5.0, 1.0, 3.0, 0.4, mdm, mdv)
    all_sym_functions[:, 72] = symmetry_function_5(5.0, 1.0, 4.0, 0.4, mdm, mdv)
    all_sym_functions[:, 73] = symmetry_function_5(5.0, 1.0, 5.0, 0.4, mdm, mdv)
    all_sym_functions[:, 74] = symmetry_function_5(5.0, 1.0, 1.0, 0.5, mdm, mdv)
    all_sym_functions[:, 75] = symmetry_function_5(5.0, 1.0, 2.0, 0.5, mdm, mdv)
    all_sym_functions[:, 76] = symmetry_function_5(5.0, 1.0, 3.0, 0.5, mdm, mdv)
    all_sym_functions[:, 77] = symmetry_function_5(5.0, 1.0, 4.0, 0.5, mdm, mdv)
    all_sym_functions[:, 78] = symmetry_function_5(5.0, 1.0, 5.0, 0.5, mdm, mdv)

    return all_sym_functions
