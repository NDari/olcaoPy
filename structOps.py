# This library contains the various functions which are used to carry out the
# various operations needed in OLCAO and EMU. The functions defined here are
# concerned with the operations need to to manipulate an atomic structure, and
# to extract various information from that structure. 
#
# version 0.x - 1.0, 2013-2014, Naseer A. Dari, Computational Physics Group.
# University of Missouri - Kansas City

# import the modules needed in these functions.
import numpy as np
import fileOps as fo
import constants as co
import math as m

def makeRealLattice(cellInfo):
    """
    This function creats the real lattice matrix from the magnitudes of the
    a, b, and c vectors contained in a typical olcao.skl file.
    """
    # lets redefine the passed parameters using the physical names. This is
    # wasteful, but makes the code more readable.
    a = cellInfo[0]
    b = cellInfo[1]
    c = cellInfo[2]
    # Convert the angles to radians.
    alpha = m.radians(cellInfo[3])
    beta  = m.radians(cellInfo[4])
    gamma = m.radians(cellInfo[5])
    # Start the construction of the RLM, the real lattice array.
    rlm = np.zeros(shape=(3,3), dtype=float)
    # Assume that a and x are coaxial.
    rlm[0][0] = a
    rlm[0][1] = 0.0
    rlm[0][2] = 0.0
    # b is then in the xy-plane.
    rlm[1][0] = (b * m.cos(gamma))
    rlm[1][1] = (b * m.sin(gamma))
    rlm[1][2] = 0.0
    # c is a mix of x,y, and z directions.
    rlm[2][0] = (c * m.cos(beta))
    rlm[2][1] = (c * (m.cos(alpha) - m.cos(gamma)*m.cos(beta)) / m.sin(gamma))
    rlm[2][2] = (c * m.sqrt(1.0 - m.cos(beta)**2 - ((rlm[2][1]/c)**2)))
    # now lets correct for numerical errors.
    rlm[rlm < 0.000000001] = 0.0
    return rlm

def makeRealLatticeInv(cellInfo):
    """
    This function inverts the real lattice matrix, which is created from the 
    magnitude of the cell vectors and the angles read in from the olcao.skl
    file.
    """
    rlm = makeRealLattice(cellInfo)
    # create the inverse real lattice matrix, and fill it.
    invRlm = np.zeros(shape=(3,3), dtype=float)
    invRlm = np.linalg.inv(rlm)
    return invRlm

def fracToCart(coors, rlm):
    """
    This function converts the fraction abc coordinates of atoms, passed in the
    array "coors" with the real lattice matrix passed in "rlm" into cartesian 
    coordinates in real space.
    """
    fracCoors = np.zeros(shape=(len(coors),3), dtype=float)
    for atom in xrange(len(coors)):
        fracCoors[atom, 0] = sum(coors[atom, :] * rlm[:, 0]) 
        fracCoors[atom, 1] = sum(coors[atom, :] * rlm[:, 1])  
        fracCoors[atom, 2] = sum(coors[atom, :] * rlm[:, 2])
    
    return fracCoors

def cartToFrac(coors, invRlm):
    """
    This function converts the cartesian xyz coordinates of atoms, passed in the
    array "coors" with the inverse real lattice matrix passed in "invRlm" into 
    fractional coordinates in real space.
    """
    cartCoors = np.zeros(shape=(len(coors),3), dtype=float)
    for atom in xrange(len(coors)):
        cartCoors[atom, 0] = sum(coors[atom, :] * invRlm[:, 0])  
        cartCoors[atom, 1] = sum(coors[atom, :] * invRlm[:, 1])  
        cartCoors[atom, 2] = sum(coors[atom, :] * invRlm[:, 2])  
    
    return cartCoors

def dist(x, y):
    """
    This function computes the distance between two atoms, who's location is
    given in cartesian coordinates.
    """
    return m.sqrt(sum((x[:]-y[:])*(x[:]-y[:])))

def minDistMat(coors, rlm, invRlm):
    """
    This function creates the min. distance matrix between all the points in the
    systems who's cartesian coordinates of are passed to this function
    in coors. the points here may refer to the positions of atoms, or
    the positions of mesh grid points, or anything else.
    The min. distance here is the distance between two points when the periodic
    boundary conditions (PBCs) are taken into account. For example, consider the 
    (silly) 1D system:

        [ABCDEFG]

    where the distance between A and G is normally calculated across the BCDEF 
    atoms. However, when PBCs are considered, they are immediately connected since
        
        ABCDEFGABCDEFG....

    The rlm argument is the real lattice matrix, and invRlm is its inverse.
    """
    numPts  = len(coors)
    f       = np.zeros(shape=(numPts, numPts), dtype=float)
    f[:, :] = 1000000000.0 

    for x in xrange(-1, 2):
        for y in xrange(-1, 2):
            for z in xrange(-1, 2):
                e = cartToFrac(coors, invRlm)
                for pt in xrange(numPts):
                    e[pt, :] += [x, y, z]
                e = fracToCart(e, rlm)
                for pt1 in xrange(numPts-1):
                    for pt2 in xrange(pt1+1, numPts):
                        val = dist(e[pt1, :], coors[pt2, :])
                        if val < f[pt1,pt2]:
                            f[pt1, pt2] = val
                            f[pt2, pt1] = val
    for i in xrange(len(f)):
        f[i, i] = 0.0
    return f

def getSysCovRads(aNames):
    """
    This function returns an array of covalant radii for the atoms in the system.
    The names of the atoms is passed to this function
    """
    numAtoms = len(aNames)
    covRads = np.zeros(shape=(numAtoms), dtype=float)
    covRads[:] = co.covalRad[aNames[:]]
    return covRads

def getBondingList(covRads, mdm, bf = 1.1):
    """
    This function creates a 2D list of bonds. for every atom x, we get a list of 
    atoms to which x is bonded.
    The passed arguments are "covRads", which is a list of the covalent radii for
    every atom, and "mdm" which is the min. distance between atoms, when
    PBCs are taken into consideration. the "bf" argument is
    is the bonding factor. atoms are considered to be bonded if their distance is
    less or equal to the sum of their covalent radii * bf. The default for bf
    is 1.1.
    NOTE: The returned list starts at 0. so atom 102 will be returned
    as atom 101, etc.
    """
    numAtoms = covRads.size
    bondingList = np.zeros(shape=(numAtoms), dtype = object)
    for atom1 in range(numAtoms):
        atomBonds = []
        for atom2 in range(numAtoms):
            if atom1 != atom2:
                bondDist = (covRads[atom1] + covRads[atom2]) * bf
                if bondDist >= mdm[atom1][atom2]:
                    atomBonds.append(atom2)
        bondingList[atom1] = atomBonds

    return bondingList

def getBondingLengthList(covRads, mdm, bf = 1.1):
    """
    This function creates a 2D list of bonds. for every atom x, we get a list of 
    bond lengths of atoms to which x is bonded.
    The passed arguments are "covRads", which is a list of the covalent radii for
    every atom, and "mdm" which is the min. distance between atoms, when
    PBCs are taken into consideration. the "bf" argument is the bonding factor. 
    atoms are considered to be bonded if their distance is less or equal 
    to the sum of their covalent radii * bf. The default for bonding
    factor is 1.1.
    """
    numAtoms = covRads.size
    bondingLengthList = np.zeros(shape=(numAtoms), dtype = object)
    for atom1 in range(numAtoms):
        atomBonds = []
        for atom2 in range(numAtoms):
            if atom1 != atom2:
                bondDist = (covRads[atom1] + covRads[atom2]) * bf
                if bondDist >= mdm[atom1][atom2]:
                    atomBonds.append(mdm[atom1][atom2])
        bondingLengthList[atom1] = atomBonds

    return bondingLengthList

def printBLStruct(bf = 1.1):
    """
    This function prints the bonding list of a given structure contained in the
    structure.dat file.
    """
    st = fo.readFile("structure.dat")
    aNames = fo.aNamesStruct(st)
    covRads = getSysCovRads(aNames)
    coors = fo.coorsStruct(st)
    rlm = fo.cellVecsStruct(st)
    mlr = np.linalg.inv(rlm)
    mdm = minDistMat(coors, rlm, mlr)
    bondingList = getBondingList(covRads, mdm, bf)
    bondingLengthList = getBondingLengthList(covRads, mdm, bf)
    numAtoms = covRads.size
    string = ""
    for atom in range(numAtoms):
        numBonds = len(bondingList[atom])
        string = string + "Atom:\t" + str(atom+1) + "  " 
        string = string + "numBonds:\t" + str(numBonds) + "\n"
        for bond in range(numBonds):
            string = string + str(bondingList[atom][bond]+1) + ":"
            string = string + str(bondingLengthList[atom][bond]) + " "
        string += "\n"
    f = open("bondingList" , 'w')
    f.write(string)
    f.close
    return

def getBondAngleList(covRads, mdm, coors,  bf = 1.1):
    """
    This function returns the bond angles for an atom. a bond angle is the angle
    between a given set of two bonds for an atom.
    The passed parameters are the covalant radii of the atoms in the system, the
    coordinates (cartesian) of the atoms in the system, minimum distance matrix,
    and the bonding factor.
    """
    # first, get the number of atoms.
    numAtoms = covRads.size
    # get bonding list.
    BL = getBondingList(covRads, mdm, bf)
    # create a numpy array for the bond angle list.
    bondAngleList = np.zeros(shape=(numAtoms), dtype = object)
    # create 3 numpy arrays for the positions of each 3 atom set.
    pos1 = np.zeros(shape=(3), dtype=float)
    pos2 = np.zeros(shape=(3), dtype=float)
    pos3 = np.zeros(shape=(3), dtype=float)
    # create 2 numpy arrays for the vectors from atom 1 to 2, and atom 1 to 3.
    vec12 = np.zeros(shape=(3), dtype=float)
    vec13 = np.zeros(shape=(3), dtype=float)
    for atom1 in range(numAtoms):
        i = 0
        # get coordinates of atom 1.
        pos1 = coors[atom1]
        numBonds = len(BL[atom1])
        # number of possible angles is numBonds choose 2.
        numAngles = (m.factorial(numBonds)/(2*m.factorial(numBonds-2)))
        angleList = np.zeros(shape=(numAngles), dtype=float)
        for atom2 in range(numBonds):
            pos2 = coors[BL[atom1][atom2]]
            # find the vector from atom 1 to 2.
            vec12 = pos2[:] - pos1[:]
            atom3 = atom2 + 1
            while (atom3 < numBonds):
                # get position of the 3rd atom.
                pos3 = coors[BL[atom1][atom3]]
                # get vector from 1 to 3.
                vec13 = pos3[:] - pos1[:]
                angleList[i] = m.acos((vec12.dot(vec13))/(
                                np.linalg.norm(vec12) * np.linalg.norm(vec13)))
                angleList[i] = m.degrees(angleList[i])
                i += 1
                atom3 += 1
        bondAngleList[atom1] = angleList
    return bondAngleList
                
def printBAStruct(bf = 1.1):
    """
    This function created the bond angle list and prints it to a file.
    """
    st = fo.readFile("structure.dat")
    aNames = fo.aNamesStruct(st)
    covRads = getSysCovRads(aNames)
    coors = fo.coorsStruct(st)
    rlm = fo.cellVecsStruct(st)
    mlr = np.linalg.inv(rlm)
    mdm = minDistMat(coors, rlm, mlr)
    bondingList = getBondingList(covRads, mdm, bf)
    numAtoms = covRads.size
    bondAngleList = getBondAngleList(covRads, mdm, coors, bf)
    string = ""
    for atom in range(numAtoms):
        i = 0
        numBonds = len(bondingList[atom])
        string = string + "Atom:\t" + str(atom+1) + "  "
        string = string + "NumAngles:\t" + str(len(bondAngleList[atom])) + "\n"
        for atom1 in range(numBonds):
            atom2 = atom1 + 1
            while (atom2 < numBonds):
                string = string + str(bondingList[atom][atom1]+1) + ":"
                string = string + str(bondingList[atom][atom2]+1) + ":"
                string = string + str(bondAngleList[atom][i]) + " "
                i += 1
                atom2 += 1
        string += "\n"
    f = open("bondAngles", 'w')
    f.write(string)
    f.close
    return

def getEnvList(numAtoms, mdm, cutOff):
    """
    This function returns a list of atoms who are within a cutoff distance from
    each atom.
    The arguments are the number of atoms, the minimal distance matrix for this
    system (calculated by mdm function), and a cutoff distance.
    """
    envList = np.zeros(shape=(numAtoms), dtype = object)
    for atom1 in range(numAtoms):
        atomEnv = []
        for atom2 in range(numAtoms):
            if atom1 != atom2:
                if cutOff >= mdm[atom1][atom2]:
                    atomEnv.append(atom2)
        envList[atom1] = atomEnv

    return envList

def getEnvLengthsList(numAtoms, mdm, cutOff):
    """
    This function returns a list of distances of atoms who are within the 
    "environment of every atom in the system. atoms are considered to be within 
    the environment of an atom if the are within a cutoff distance from that atom.
    The arguments are the number of atoms, the minimal distance matrix for this
    system (calculated by minDistMat function), and a cutoff distance.
    """
    envLengthsList = np.zeros(shape=(numAtoms), dtype = object)
    for atom1 in range(numAtoms):
        atomEnv = []
        for atom2 in range(numAtoms):
            if atom1 != atom2:
                if cutOff >= mdm[atom1][atom2]:
                    atomEnv.append(mdm[atom1][atom2])
        envLengthsList[atom1] = atomEnv

    return envLengthsList

def getEnvAngleList(numAtoms, mdm, coors,  cutOff):
    """
    This function returns the environmental angles for an atom. the env. angles are
    the angles between any triplet in an atoms envirmentment.
    The passed parameters are the covalant radii of the atoms in the system, the
    coordinates (cartesian) of the atoms in the system, minimum distance matrix,
    and the bonding factor.
    """
    # get the envirmental list.
    EL = getEnvList(numAtoms, mdm, cutOff)
    # create a numpy array for the env. angle list.
    envAngleList = np.zeros(shape=(numAtoms), dtype = object)
    # create 3 numpy arrays for the positions of each 3 atom set.
    pos1 = np.zeros(shape=(3), dtype=float)
    pos2 = np.zeros(shape=(3), dtype=float)
    pos3 = np.zeros(shape=(3), dtype=float)
    # create 2 numpy arrays for the vectors from atom 1 to 2, and atom 1 to 3.
    vec12 = np.zeros(shape=(3), dtype=float)
    vec13 = np.zeros(shape=(3), dtype=float)
    for atom1 in range(numAtoms):
        i = 0
        # get coordinates of atom 1.
        pos1 = coors[atom1]
        numEnvAtoms = len(EL[atom1])
        # number of possible angles is numEnvAtoms choose 2.
        numAngles = (m.factorial(numEnvAtoms)/(2*m.factorial(numEnvAtoms-2)))
        angleList = np.zeros(shape=(numAngles), dtype=float)
        for atom2 in range(numEnvAtoms):
            pos2 = coors[EL[atom1][atom2]]
            # find the vector from atom 1 to 2.
            vec12 = pos2[:] - pos1[:]
            atom3 = atom2 + 1
            while (atom3 < numEnvAtoms):
                # get position of the 3rd atom.
                pos3 = coors[EL[atom1][atom3]]
                # get vector from 1 to 3.
                vec13 = pos3[:] - pos1[:]
                value = ((vec12.dot(vec13))/(
                               np.linalg.norm(vec12) * np.linalg.norm(vec13)))
                # IMPORTANT: This weirdness is implemented here to catch the
                # issues I've encountered where an error is thrown due to round-
                # off errors around values of 1.0 and -1.0. I dont think this
                # is an issue in general, but 1) I could be wrong, and 2) this
                # is not elegent. the user is invited to fix this and send me
                # the corrected version.
                if value > 1.0:
                    angleList[i] = 0.0
                elif value < -1.0:
                    angleList[i] = m.pi
                else:
                    angleList[i] = m.acos(value)
                angleList[i] = m.degrees(angleList[i])
                i += 1
                atom3 += 1
        envAngleList[atom1] = angleList
    return envAngleList

def cFn(distance, cutOff):
    """
    This function defined the cutOff function as defined in:

    "Atom-Centered symmetry functions for contructing high-dimentional neural
    network potentials", by Jorg Behler, J. Chem. Phys. 134, 074106 (2011)"

    It is important to note that the cut off function returns 0 if the distance
    between the two atoms in question is greater than some critical (cut off)
    radius. This is NOT done here, since the list of lengths passed here are 
    filtered, such that only lengths below the cutoff radius are passed to this
    function.
    """
    return (0.5 * (m.cos((m.pi*distance)/cutOff) + 1))

def genSymFn1(numAtoms, mdm, cutOff):
    """
    This function returns the first symmetry function of a system as defined in:

    "Atom-Centered symmetry functions for contructing high-dimentional neural
    network potentials", by Jorg Behler, J. Chem. Phys. 134, 074106 (2011)"

    the passed parameters are the number of atoms, the minimal distance matrix, and
    the cutoff distance as defined in the above paper
    """
    # Lets first get the environmental lengths list using the passed cutoff radius.
    ELL = getEnvLengthsList(numAtoms, mdm, cutOff)
    symFn1 = np.zeros(shape=(numAtoms), dtype=float)
    for atom1 in range(numAtoms):
        for atom2 in range(len(ELL[atom1])):
            symFn1[atom1] += cFn(ELL[atom1][atom2], cutOff)

    return symFn1

def genSymFn2(numAtoms, mdm, cutOff, rs, eta):
    """
    This function returns the second symmetry function of a system as defined in:

    "Atom-Centered symmetry functions for contructing high-dimentional neural
    network potentials", by Jorg Behler, J. Chem. Phys. 134, 074106 (2011)"

    the passed parameters are number of atoms, minimal distance matrix, the cutoff
    distance, and Rs and eta as defined in the above paper.
    """
    # Lets first get the environmental lengths list using the passed cutoff radius.
    ELL = getEnvLengthsList(numAtoms, mdm, cutOff)
    symFn2 = np.zeros(shape=(numAtoms), dtype=float)
    for atom1 in range(numAtoms):
        for atom2 in range(len(ELL[atom1])):
            symFn2[atom1] += (m.exp(-eta*((ELL[atom1][atom2] - rs)**2)) * 
                             (cFn(ELL[atom1][atom2], cutOff)))

    return symFn2 

def genSymFn3(numAtoms, mdm, cutOff, kappa):
    """
    This function returns the third symmetry function of a system as defined in:

    "Atom-Centered symmetry functions for contructing high-dimentional neural
    network potentials", by Jorg Behler, J. Chem. Phys. 134, 074106 (2011)"

    the passed parameters are the number of atoms, minimal distance matrix, the
    cutoff distance, and kappa as defined in the paper above.
    """
    # Lets first get the environmental lengths list using the passed cutoff radius.
    ELL = getEnvLengthsList(numAtoms, mdm, cutOff)
    symFn3 = np.zeros(shape=(numAtoms), dtype=float)
    for atom1 in range(numAtoms):
        for atom2 in range(len(ELL[atom1])):
            symFn3[atom1] += (m.cos(kappa * (ELL[atom1][atom2])) * 
                             (cFn(ELL[atom1][atom2], cutOff)))

    return symFn3

def genSymFn4(numAtoms, mdm, coors, cutOff, lam, zeta, eta):
    """
    This function returns the forth symmetry function of a system as defined in:

    "Atom-Centered symmetry functions for contructing high-dimentional neural
    network potentials", by Jorg Behler, J. Chem. Phys. 134, 074106 (2011)"

    the passed parameters are the number of atoms, minimum distance matrix, the
    catersian coordinates, the cutoff distance, and the parameters lam (lambda),
    zeta, and eta as defined in the above paper.
    """
    symFn4 = np.zeros(shape=(numAtoms), dtype=float)
    # get the envimental list.
    EL = getEnvList(numAtoms, mdm, cutOff)
    # get env. angle list.
    EAL = getEnvAngleList(numAtoms, mdm, coors,  cutOff)
    for atom1 in range(numAtoms):
        i = 0
        numEnvAtoms = len(EL[atom1])
        for atom2 in range(numEnvAtoms):
            dist12 = mdm[atom1][EL[atom1][atom2]]
            atom3 = atom2 + 1
            while (atom3 < numEnvAtoms):
                dist13 = mdm[atom1][EL[atom1][atom3]]
                dist23 = mdm[EL[atom1][atom2]][EL[atom1][atom3]]
                angle  = EAL[atom1][i]
                symFn4[atom1] += (((1+lam*m.cos(angle))**zeta) * 
                            (m.exp(-eta*((dist12**2) +
                            (dist13**2) + (dist23**2)))) * 
                            (cFn(dist12, cutOff) *
                             cFn(dist13, cutOff) * 
                             cFn(dist23, cutOff)))
                i += 1
                atom3 += 1
        symFn4[atom1] *= (2**(1-zeta))

    return symFn4

def genSymFn5(numAtoms, mdm, coors, cutOff, lam, zeta, eta):
    """
    This function returns the fifth symmetry function of a system as defined in:

    "Atom-Centered symmetry functions for contructing high-dimentional neural
    network potentials", by Jorg Behler, J. Chem. Phys. 134, 074106 (2011)"

    the passed parameters are the number of atoms, minimum distance matrix, the
    catersian coordinates, the cutoff distance, and the parameters lam (lambda),
    zeta, and eta as defined in the above paper.
    """
    symFn5 = np.zeros(shape=(numAtoms), dtype=float)
    # get the envimental list.
    EL = getEnvList(numAtoms, mdm, cutOff)
    # get env. angle list.
    EAL = getEnvAngleList(numAtoms, mdm, coors,  cutOff)
    for atom1 in range(numAtoms):
        i = 0
        numEnvAtoms = len(EL[atom1])
        for atom2 in range(numEnvAtoms):
            dist12 = mdm[atom1][EL[atom1][atom2]]
            atom3 = atom2 + 1
            while (atom3 < numEnvAtoms):
                dist13 = mdm[atom1][EL[atom1][atom3]]
                angle  = EAL[atom1][i]
                symFn5[atom1] += (((1+lam*m.cos(angle))**zeta) * 
                                  (m.exp(-eta*((dist12**2)+
                                  (dist13**2)))) * 
                                  (cFn(dist12, cutOff) *
                                   cFn(dist13, cutOff)))
                i += 1
                atom3 += 1
        symFn5[atom1] *= (2**(1-zeta))
    return symFn5

def checkThresh(arr, thrash):
    """
    This function takes a 2D array, arr, and checks if each element is less than
    the thrashhold, thrash. if so, that element is set to 0.0.
    """
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] < thrash:
                arr[i][j] = 0.0
    return arr

def printSymFns():
    """ 
    This function prints the neural network input file, generated from the 5 
    symmetry functions defined in:


    
    The parameters used are as follows (note that we use Angstroms as apposed to
    the Bohrs used in the above paper):

    G1 : Rc = 1 to 5, in 0.5 increments.
    G2 : Rs = 0.5 to 5 in 0.5 increments, and Rc = 5, eta = 12.
    G3 : kappa = 1 to 5, in 1 increments, and Rc = 5
    G4 : zeta = 1, 2, 4, 16, 64, and Rc = 5, and eta = 12, lambda = 1
    G4 : same as G4.
    """
    # lets begin by getting all the imformation we need. First, read the file...
    sdat = fo.readFile("structure.dat")
    numAtoms = fo.SdatNumAtoms(sdat)
    coors = fo.SdatCoors(sdat)
    rlm = fo.SdatCellVecs(sdat)
    mlr = np.linalg.inv(rlm)
    mdm = minDistMat(coors, rlm, mlr)
    allSym = np.zeros(shape=(numAtoms,78), dtype=float)
    # start with G1. 
    allSym[:, 0] = genSymFn1(numAtoms, mdm, 0.5)
    allSym[:, 1] = genSymFn1(numAtoms, mdm, 0.7)
    allSym[:, 2] = genSymFn1(numAtoms, mdm, 0.9)
    allSym[:, 3] = genSymFn1(numAtoms, mdm, 1.1)
    allSym[:, 4] = genSymFn1(numAtoms, mdm, 1.3)
    allSym[:, 5] = genSymFn1(numAtoms, mdm, 1.5)
    allSym[:, 6] = genSymFn1(numAtoms, mdm, 1.7)
    allSym[:, 7] = genSymFn1(numAtoms, mdm, 1.9)
    allSym[:, 8] = genSymFn1(numAtoms, mdm, 2.1)
    allSym[:, 9] = genSymFn1(numAtoms, mdm, 2.3)
    allSym[:,10] = genSymFn1(numAtoms, mdm, 2.5)
    allSym[:,11] = genSymFn1(numAtoms, mdm, 2.7)
    allSym[:,12] = genSymFn1(numAtoms, mdm, 2.9)
    allSym[:,13] = genSymFn1(numAtoms, mdm, 3.1)
    allSym[:,14] = genSymFn1(numAtoms, mdm, 3.3)
    allSym[:,15] = genSymFn1(numAtoms, mdm, 3.5)
    allSym[:,16] = genSymFn1(numAtoms, mdm, 3.7)
    allSym[:,17] = genSymFn1(numAtoms, mdm, 3.9)
    allSym[:,18] = genSymFn1(numAtoms, mdm, 4.1)
    allSym[:,19] = genSymFn1(numAtoms, mdm, 4.3)
    allSym[:,20] = genSymFn1(numAtoms, mdm, 4.5)
    allSym[:,21] = genSymFn1(numAtoms, mdm, 4.7)
    allSym[:,22] = genSymFn1(numAtoms, mdm, 4.9)
    allSym[:,23] = genSymFn1(numAtoms, mdm, 5.1)
    allSym[:,24] = genSymFn1(numAtoms, mdm, 5.3)
    allSym[:,25] = genSymFn1(numAtoms, mdm, 5.5)
    allSym[:,26] = genSymFn1(numAtoms, mdm, 5.7)
    # now G2.
    allSym[:,27] = genSymFn2(numAtoms, mdm, 5.0, 2.0, 12)
    allSym[:,28] = genSymFn2(numAtoms, mdm, 5.0, 2.2, 12)
    allSym[:,29] = genSymFn2(numAtoms, mdm, 5.0, 2.4, 12)
    allSym[:,30] = genSymFn2(numAtoms, mdm, 5.0, 2.6, 12)
    allSym[:,31] = genSymFn2(numAtoms, mdm, 5.0, 2.8, 12)
    allSym[:,32] = genSymFn2(numAtoms, mdm, 5.0, 3.0, 12)
    allSym[:,33] = genSymFn2(numAtoms, mdm, 5.0, 3.2, 12)
    allSym[:,34] = genSymFn2(numAtoms, mdm, 5.0, 3.4, 12)
    allSym[:,35] = genSymFn2(numAtoms, mdm, 5.0, 3.6, 12)
    allSym[:,36] = genSymFn2(numAtoms, mdm, 5.0, 3.8, 12)
    allSym[:,37] = genSymFn2(numAtoms, mdm, 5.0, 4.0, 12)
    allSym[:,38] = genSymFn2(numAtoms, mdm, 5.0, 4.2, 12)
    allSym[:,39] = genSymFn2(numAtoms, mdm, 5.0, 4.4, 12)
    allSym[:,40] = genSymFn2(numAtoms, mdm, 5.0, 4.6, 12)
    allSym[:,41] = genSymFn2(numAtoms, mdm, 5.0, 4.8, 12)
    allSym[:,42] = genSymFn2(numAtoms, mdm, 5.0, 5.0, 12)
    # and G3.
    allSym[:,43] = genSymFn3(numAtoms, mdm, 5.0, 1.0)
    allSym[:,44] = genSymFn3(numAtoms, mdm, 5.0, 1.5)
    allSym[:,45] = genSymFn3(numAtoms, mdm, 5.0, 2.0)
    allSym[:,46] = genSymFn3(numAtoms, mdm, 5.0, 2.5)
    allSym[:,47] = genSymFn3(numAtoms, mdm, 5.0, 3.0)
    # now G4.
    allSym[:,48] = genSymFn4(numAtoms, mdm, coors, 5.0, 1.0,  1.0,   0.1)
    allSym[:,49] = genSymFn4(numAtoms, mdm, coors, 5.0, 1.0,  2.0,   0.1)
    allSym[:,50] = genSymFn4(numAtoms, mdm, coors, 5.0, 1.0,  4.0,   0.1)
    allSym[:,51] = genSymFn4(numAtoms, mdm, coors, 5.0, 1.0, 16.0,   0.1)
    allSym[:,52] = genSymFn4(numAtoms, mdm, coors, 5.0, 1.0, 64.0,   0.1)
    allSym[:,53] = genSymFn4(numAtoms, mdm, coors, 5.0, 1.0,  1.0,  0.01)
    allSym[:,54] = genSymFn4(numAtoms, mdm, coors, 5.0, 1.0,  2.0,  0.01)
    allSym[:,55] = genSymFn4(numAtoms, mdm, coors, 5.0, 1.0,  4.0,  0.01)
    allSym[:,56] = genSymFn4(numAtoms, mdm, coors, 5.0, 1.0, 16.0,  0.01)
    allSym[:,57] = genSymFn4(numAtoms, mdm, coors, 5.0, 1.0, 64.0,  0.01)
    allSym[:,58] = genSymFn4(numAtoms, mdm, coors, 5.0, 1.0,  1.0, 0.001)
    allSym[:,59] = genSymFn4(numAtoms, mdm, coors, 5.0, 1.0,  2.0, 0.001)
    allSym[:,60] = genSymFn4(numAtoms, mdm, coors, 5.0, 1.0,  4.0, 0.001)
    allSym[:,61] = genSymFn4(numAtoms, mdm, coors, 5.0, 1.0, 16.0, 0.001)
    allSym[:,62] = genSymFn4(numAtoms, mdm, coors, 5.0, 1.0, 64.0, 0.001)
    # finally G5.
    allSym[:,63] = genSymFn5(numAtoms, mdm, coors, 5.0, 1.0,  1.0,   0.1)
    allSym[:,64] = genSymFn5(numAtoms, mdm, coors, 5.0, 1.0,  2.0,   0.1)
    allSym[:,65] = genSymFn5(numAtoms, mdm, coors, 5.0, 1.0,  4.0,   0.1)
    allSym[:,66] = genSymFn5(numAtoms, mdm, coors, 5.0, 1.0, 16.0,   0.1)
    allSym[:,67] = genSymFn5(numAtoms, mdm, coors, 5.0, 1.0, 64.0,   0.1)
    allSym[:,68] = genSymFn5(numAtoms, mdm, coors, 5.0, 1.0,  1.0,  0.01)
    allSym[:,69] = genSymFn5(numAtoms, mdm, coors, 5.0, 1.0,  2.0,  0.01)
    allSym[:,70] = genSymFn5(numAtoms, mdm, coors, 5.0, 1.0,  4.0,  0.01)
    allSym[:,71] = genSymFn5(numAtoms, mdm, coors, 5.0, 1.0, 16.0,  0.01)
    allSym[:,72] = genSymFn5(numAtoms, mdm, coors, 5.0, 1.0, 64.0,  0.01)
    allSym[:,73] = genSymFn5(numAtoms, mdm, coors, 5.0, 1.0,  1.0, 0.001)
    allSym[:,74] = genSymFn5(numAtoms, mdm, coors, 5.0, 1.0,  2.0, 0.001)
    allSym[:,75] = genSymFn5(numAtoms, mdm, coors, 5.0, 1.0,  4.0, 0.001)
    allSym[:,76] = genSymFn5(numAtoms, mdm, coors, 5.0, 1.0, 16.0, 0.001)
    allSym[:,77] = genSymFn5(numAtoms, mdm, coors, 5.0, 1.0, 64.0, 0.001)
    string = ""
    for atom in range(numAtoms):
        for item in range(78):
            string += str(allSym[atom][item])
            string += " "
        string += "\n"
    f = open("symFns", 'w')
    f.write(string)
    f.close
    return
