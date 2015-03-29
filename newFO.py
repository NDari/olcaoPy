# This package contains all the operations involving reading files, reading
# lines, writing files, and extracting information from specific files 
# encountered in an OLCAO job. The overall idea is to be as general as 
# possible, while also providing specific functions which work with specific
# files. For instance, it is common to wish to extract the number of atoms of
# a system under study. This information is contained in many locations, so
# we will have specific functions to extract it from olcao.skl file, and
# structure.dat file, and so on. This makes the code very specific, and not
# pretty. However, a user may wish to use these low level functions and
# wrap them in a higher level function that get the atomic coordinates
# by looking at the file it is instructed to read (either olcao.skl or
# structure.dat) and call the right low level function.
# the idea is for this package (and other packages delevoped so far) is
# to be extended by users, as they encounter different files, and need to
# exctract different things from them. thus we will have one repository
# in this packages to reduce duplicate efforts.
# NOTE: you can go between the various sections (or file types) in this 
# code by searching for the "###" tag.

# version 0.x - 1.0 2013-2014 Naseer A. Dari, Computational Physics Group.
# University of Missouri - Kansas City

# import needed modules
import re
import sys
import olcaoPy.constants

### Generic functions.

# Let's define a function to read a file returning a 2D array.

def readFile(fileName, splitter = '\s+'):
    """
    This function reads a file, returning a 2D array A(x,y)

    x > line number
    y > element number

    where the elements in the line are created by splitting the line
    according to the splitter argument. The default values for the splitter
    is one or more spaces.
    """
    array = []
    f = open(fileName, 'r')

    # hold all the lines in the file, by reading all of it, and
    # stripping the newline at the end of each line.
    lines = [line.strip() for line in f.readlines()]
    for line in lines:
        # split each line using the pslitter, and append it to our
        # 2D array.
        array.append(re.split(splitter, line))
    f.close()
    return array

# Let's define a function to write a file from a 2D array.

def writeFloats(fileName, fileArr, splitter = ' '):
    """
    This function writes a file with name fileName, from a 2D array. The
    splitter argument is what is to be inserted in between the elements
    of the array, and its default is a space. all elements in the array
    passed are automatically converted to strings.
    """
    f = open(fileName, 'w')

    string = ""

    for line in range(len(fileArr)):
        # first, convert all the words to strings.
        l = [str(word) for word in fileArr[line]]
        string += splitter.join(l)

        # add the newline before next line
        string += "\n"

    f.write(string)
    f.close()
    return

# Let's define a function that reads a line from a file and splits it, 
# returning a 1D array.

def prepLine(fileName, line, splitter = '\s+'):
    """
    This function takes a fileName, and reads the next line, then splits it by
    the passed splitter, returning a 1D array. The default for the splitter is
    one or more spaces.
    If file's name is passed as "", then the line (or object to be split) must
    be passed.
    """
    values = []
    if fileName != "" and line == "":
        line = fileName.readline()
        line.strip()
        values = re.split(splitter, line)
        values.pop()
    elif fileName == "" and line != "":
        values = re.split(splitter, line)
    else:
        sys.exit("You must either pass the file or the line to be split." +
                "\n" + "Dont Know what to do. Aborting script.")
        return values

### olcao.skl file functions.

# Now lets define functions which return various information contained
# in olcao.skl, such as the title, number of atoms, etc.

def SklTitle(skl):
    """
    This function returns the title contained in an array which was created
    by reading the olcao.skl file.
    """
    title = ""
    # skip first line with label "title"
    for i in range(1, len(skl)): 
        # read until the label "end".
        if (skl[i][0] != 'end'): 
            title += " ".join(skl[i])
            title += "\n" 
        else:
            # strip the last (extra) newline, and exit.
            title = title.rstrip('\n') 
            break
    return title

def SklCellInfo(skl):
    """
    This function returns the cell vectors a, b, and c as well as the cell
    angles alpha, beta, and gamma in a 1D array:
    [a, b, c, alpha, beta, gamma]
    """
    cellInfo = []
    # Find the line in skl containing the file, which starts with cell
    for i in range(len(skl)):
        if (skl[i][0] == 'cell'): 
            # the angles are in the next line.
            cellInfo = [float(x) for x in skl[i+1]]
            break
    return cellInfo

def SklCoordType(skl):
    """
    This function returns the coordinate type used in an array which was
    created from reading the olcao.skl file. The coordinate type returned is
    either "F" for fractional, or "C" for cartesian.
    """
    # Find the line in skl containing the file, which starts with cell
    for i in range(len(skl)):
        if (skl[i][0] == 'cell'): 
            # coord. type is contained in two lines after that. 
            if 'frac' in skl[i+2][0]:
                return 'F'
            elif 'cart' in skl[i+2][0]:
                return 'C'
            else:
                sys.exit("Unknow coordinate type: " + skl[i+2][0])

def SklNumAtoms(skl):
    """
    This function extracts the number of atoms from an array which was created
    from reading the olcao.skl file. 
    """
    # Find the line in skl containing the file, which starts with cell
    for i in range(len(skl)):
        if (skl[i][0] == 'cell'): 
            # the number of atoms is two lines after that.
            return int(skl[i+2][1]) 

def SklCoors(skl):
    """
    This function returns the coordinates of the atoms contained in an array
    which was created from reading the olcao.skl file. 
    """
    a = 0
    coors = []
    # Find the line in skl containing the file, which starts with cell
    for i in range(len(skl)):
        if (skl[i][0] == 'cell'): 
            # the number of atoms is two lines after that.
            numAtoms = int(skl[i+2][1]) 
            # The list of coordinates starts 3 lines after "cell".
            a = i + 3 
            break
    for i in range(numAtoms):
        dummy = []
        for j in range(3):
            dummy.append(float(skl[a+i][1+j]))
        coors.append(dummy)
    return coors

def SklAtomNames(skl):
    """
    This function returns the atomic names contained in an arraywhich was 
    created from reading the olcao.skl file.
    """
    a = 0
    aNames = []
    for i in range(len(skl)):
        if (skl[i][0] == 'cell'): 
            # the number of atoms is two lines after that.
            numAtoms = int(skl[i+2][1]) 
            # The list of coordinates starts 3 lines after "cell".
            a = i + 3 
            break
    for i in range(numAtoms):
        name      = skl[a+i][0]

        # make sure that the name starts with a lower case letter. this is
        # the convension used throughout olcao.
        name      = name[0].lower() + name[1:]
        aNames.append(name)
    return aNames

def SklSpaceGrp(skl):
    """
    This function returns the space group number contained in an array which
    was created from reading the olcao.skl file. 
    """
    return skl[-3][1] # space group number is 3rd from bottom.

def SklSupercell(skl):
    """
    This function returns an array corresponding to the supercell contained in
    an array created from reading the olcao.skl file. 
    """
    supercell = []
    # supercell info is 2nd from bottom.
    supercell.append(int(skl[-2][1] ))
    supercell.append(int(skl[-2][2] ))
    supercell.append(int(skl[-2][3] ))
    return supercell

def SklSupercellMirror(skl):
    """
    This function returns an array corresponding to the supercell mirror
    contained in an array created from reading the olcao.skl file. 
    """
    supercellMirror = []
    # supercell info is 2nd from bottom. note that this information is
    # optional, so it may not be present. if not, we simply return the
    # mirror array as defined ([0, 0, 0]).
    if len(skl[-2]) > 4:
        supercellMirror.append(int(skl[-2][4]))
        supercellMirror.append(int(skl[-2][5]))
        supercellMirror.append(int(skl[-2][6]))
    else:
        supercellMirror.append(int(0))
        supercellMirror.append(int(0))
        supercellMirror.append(int(0))

    return supercellMirror

def SklCellType(skl):
    """
    This function returns the cell type for the olcao.skl file, which is
    typically either "full" or "prim".
    """
    if skl[-1][0] == "full":
        return "F"
    elif skl[-1][0] == "prim":
        return "P"
    else:
        sys.exit("Uknown cell type: " + skl[-1][0])

### .xyz file functions.

def XyzNumAtoms(xyz):
    '''
    This function returns the number of atoms in an xyz file that was read 
    with the readFile function.
    '''
    return int(xyz[0][0])

def XyzComment(xyz):
    '''
    This function returns the comment line in an xyz file.
    '''
    return " ".join(xyz[1])

def XyzAtomNames(xyz):
    '''
    This function returns the atomic names in an xyz file
    '''
    numAtoms = int(xyz[0][0])
    atomNames = []
    for i in range(numAtoms):
        name = xyz[i+2][0] # skip 2 header lines.
        name = name[0].lower() + name[1:] # olcao convention
        atomNames.append(name)
    return atomNames

def XyzCoors(xyz):
    '''
    This function returns the atomic coordinates in an xyz file
    '''
    numAtoms = int(xyz[0][0])
    coors = []
    for i in range(numAtoms):
        dummy = []
        for j in range(3):
            dummy.append(float(xyz[i+2][1+j]))
        coors.append(dummy)
    return coors

### scfV.dat file functions.

# Lets now define some functions to return the various information contained
# in the scfV.dat file.
#
# IMPORTANT NOTE: Due to the similarity between the scfV.dat and the
# gs_scfV-xx.dat file (xx being the basis type used (mb, fb, eb)) all of these
# functions will work with an array created from reading the gs_scfV-xx.dat
# file. HOWEVER, that array MUST be created, and fed to these functions. 

def ScfvNumTypes(scfv):
    """
    This function returns the number of distinct types in the system, contained
    in an array which was created from reading the scfV.dat file. 
    """
    return int(scfv[0][1])

def ScfvNumTermsPerType(scfv):
    """
    This function returns the number of terms for each unique type in the system
    contained in an array which was created from reading the scfV.dat file or the
    gs_ equavalent file.
    """
    # get the number of unique types.
    nTypes =  int(scfv[0][1])
    termsPerType = []
    # the fotmat of the file is as follows: first line (line 0) contains the # of
    # types, and the second line is the spin tag.
    # Next line contains the number of terms per type 1 which we want to
    # store, say x. then we will have x number of lines containing the actual
    # terms, then the number of terms for type 2 is printed... so we skip 2 line
    # at the top, starting at the third line.
    line = 2
    for j in range(nTypes):
        termsPerType.append(int(scfv[line][0]))
        # now lets skip that many lines+1, read the number of terms for the next
        # type, if another type exists.
        line += termsPerType[j] + 1
    return termsPerType

def ScfvPotCoeffs_up(scfv):
    """
    This function returns the A coefficients for each unique type, as a 2-D array
    A[x][y] where

    x > type number
    y > coeffecient number

    A is contructed from an array passed to this function which is in turn
    contrusted from reading the scfV.dat file. 
    
    This function returns the coeffcients with spin UP. these are used as 
    THE coefficients for exchange-correlation functionals that are spinless. 
    """
    # get the number of unique types.
    nTypes =  int(scfv[0][1])
    # number of coefficients start at 3rd line
    i = 2
    coeffs = []
    for j in range(nTypes):
        nTerms = int(scfv[i][0])
        currCoeffs = []
        i += 1
        for k in range(nTerms):
            # the coefficients are the 1st entry in these lines.
            currCoeffs.append(float(scfv[i][0]))
            # skip a line for the next coefficient
            i += 1
        coeffs.append(currCoeffs)
    return coeffs

def ScfvPotCoeffs_dn(scfv):
    """
    This function returns the A coefficients for each unique type, as a 2-D array
    A[x][y] where

    x > type number
    y > coeffecient number

    A is contructed from an array passed to this function which is in turn
    contrusted from reading the scfV.dat file. 
    
    This function returns the coeffcients with spin DOWN. these are used only
    with exchange-correlation functionals that include spin.
    """
    # get the number of unique types.
    nTypes =  int(scfv[0][1])
    # first find the line in which the spin down coefficints are
    # listed
    for j in range(len(scfv)):
        if scfv[j][0] == "SPIN_DN":
            i = j
            break
    coeffs = []
    for j in range(nTypes):
        nTerms = int(scfv[i][0])
        currCoeffs = []
        i += 1
        for k in range(nTerms):
            # the coefficients are the 1st entry in these lines.
            currCoeffs.append(float(scfv[i][0]))
            # skip a line for the next coefficient
            i += 1
        coeffs.append(currCoeffs)
    return coeffs

def ScfvPotCoeffs(scfv):
    """
    This function returns the A coefficients for each unique type, as a 3-D array
    A[x][y][z] where

    x > type number
    y > 0 for spin up, 1 for spin down
    z > coeffecient number

    A is contructed from an array passed to this function which is in turn
    contrusted from reading the scfV.dat file. 
    
    """
    # get the number of unique types.
    nTypes =  int(scfv[0][1])
    coeffs = []
    # get the spin up items coefficients, and the spin down coefficients.
    spinUp = ScfvPotCoeffs_up(scfv)
    spinDn = ScfvPotCoeffs_dn(scfv)
    for i in range(nTypes):
        totalSpin = []
        totalSpin.append(spinUp[i])
        totalSpin.append(spinDn[i])
        coeffs.append(totalSpin

    return coeffs

def ScfvPotAlphas(scfv):
    """
    This function returns the alphas for each unique type, as a 2-D array
    A[x][y] where

    x > type number
    y > alpha number

    A is contructed from an array passed to this function which is in turn
    contrusted from reading the scfV.dat file. 
    """
    # get the number of unique types.
    nTypes =  int(scfv[0][1])
    # number of alphas start at 3rd line
    i = 2
    alphas = []
    for j in range(nTypes):
        nTerms = int(scfv[i][0])
        currAlphas = []
        i += 1
        for k in range(nTerms):
            # the alphas are the 1st entry in these lines.
            currAlphas.append(float(scfv[i][1]))
            # skip a line for the next coefficient
            i += 1
        alphas.append(currAlphas)
    return alphas

def ScfvFullRhos(scfv):
    """
    This function returns the full rho for each unique type, as a 2-D array
    A[x][y] where

    x > type number
    y > full rho

    A is contructed from an array passed to this function which is in turn
    contrusted from reading the scfV.dat file. 
    """
    # get the number of unique types.
    nTypes =  int(scfv[0][1])
    # number of alphas start at 3rd line
    i = 2
    fRhos = []
    for j in range(nTypes):
        nTerms = int(scfv[i][0])
        currRhos = []
        i += 1
        for k in range(nTerms):
            # the alphas are the 1st entry in these lines.
            currRhos.append(float(scfv[i][2]))
            # skip a line for the next coefficient
            i += 1
        fRhos.append(currRhos)
    return alphas

def ScfvPartRhos(scfv):
    """
    This function returns the A coefficients for each unique type, as a 2-D array
    A[x][y] where

    x > type number
    y > partial rho number

    A is contructed from an array passed to this function which is in turn
    contrusted from reading the scfV.dat file. If such array is not passed, it
    is contructed by reading that file.
    """
    # get the number of unique types.
    nTypes = ScfvNumTypes(scfv)  
    # get num terms per each type.
    termsPerType = ScfvNumTermsPerType(scfv) 
    # coefficients start at 4th line (line 3)
    i = 3
    # create a numpy array of objects, where each object is a array of
    # partial rhos. thus creating the desired 2D matrix as noted above.
    partRhos = np.zeros(shape=(nTypes), dtype = object)
    for j in range(nTypes):
        currRhos = np.zeros(shape=(termsPerType[j]))
        for k in range(len(currRhos)):
            # the partial rhos are the 4th entry in these lines.
            currRhos[k] = scfv[i][3]
            # skip a line for the next partial rho
            i += 1
        partRhos[j] = currRhos
        # skip a line for the header containing num terms per type.
        i += 1
    return partRhos

### structure.dat file functions.

# Now we will define some functions with return the various information
# contained in the structure.dat file

def SdatCellVecs(sdat):
    """
    This function returns a 2-D array containing the cell vectors in the 
    Cartesian coordinates in this form:

    ax ay az
    bx by bz
    cx cy cz

    from a 2D array which is passed to this function. The passed array is
    constructed from reading the structure.dat file.
    """
    cellVecs = np.zeros(shape=(3,3))
    cellVecs[0][0:3] = sdat[1][0:3]
    cellVecs[1][0:3] = sdat[2][0:3]
    cellVecs[2][0:3] = sdat[3][0:3]

    return cellVecs

def SdatNumAtomSites(sdat):
    """
    This function returns the number of atoms contained in a 2D array contructed
    from reading the structure.dat file.
    """
    return int(sdat[5][0]) 

def SdatAtomTypes(sdat):
    """
    This function returns the types of the atoms contained in a 2D array which
    was contructed from reading the structure.dat file. 
    """
    numAtoms = SdatNumAtomSites(sdat)
    types = np.zeros(shape=(numAtoms), dtype = int)
    for atom in xrange(numAtoms):
        # the type information starts at the 8th line (line 7).
        types[atom] = sdat[7+atom][1]
    return types

def SdatAtomSites(sdat):
    """
    This function returns the atomic coordinates contained in a 2D array which
    was contructed from reading the structure.dat file. Note that the coors in
    the structure.dat file are in atomic units.
    """
    numAtoms = SdatNumAtomSites(sdat)
    coors = np.zeros(shape=(numAtoms,3))
    for atom in xrange(numAtoms):
        # the type information starts at the 8th line (line 7).
        coors[atom][0:3] = sdat[7+atom][2:5]
    return coors

def SdatAtomNames(sdat):
    """
    This function returns the element names in a structure contained in a 2D
    array contructure from reading the structure.dat file. 
    """
    numAtoms = SdatNumAtomSites(sdat)
    aNames = np.chararray(shape=(numAtoms), itemsize = 2)
    for atom in xrange(numAtoms):
        # the name information starts at the 8th line (line 7)
        # check that the list is in the correct order. die if not.
        if atom+1 != int(sdat[7+atom][0]):
            sys.exit("Atom site list out of order!")
        aNames[atom] = sdat[7+atom][5]
    return aNames

def SdatNumPotSites(sdat):
    '''
    This function returns the number of potential sites in a structure
    from the structure.dat file.
    '''
    for i in xrange(len(sdat)):
        if sdat[i][0] == "NUM_POTENTIAL_SITES":
            numPotSites = int(sdat[i+1][0])
            break
    return numPotSites

def SdatPotTypeAssn(sdat):
    '''
    This function return the pot type assignment of all the atoms in the
    system.
    '''
    numPotSites = SdatNumPotSites(sdat)
    potSiteAssn = np.zeros(shape=(numPotSites), dtype = int)
    if sdat[i][0] == "NUM_POTENTIAL_SITES":
        line = i + 3
        for pot in xrange(numPotSites):
            # the name information starts at the 8th line (line 7)
            # check that the list is in the correct order. die if not.
            if pot+1 != int(sdat[line+pot][0]):
                sys.exit("Potential site list out of order!")
            potSiteAssn[pot] = sdat[line+pot][1]
    return potSiteAssn

def SdatPotSites(sdat):
    '''
    The function returns the location of the potential sites for
    a structure, written in the structure.dat file.
    '''
    numPotSites = SdatNumPotSites(sdat)
    potSites = np.zeros(shape=(numPotSites, 3))
    for i in xrange(len(sdat)):
        if sdat[i][0] == "NUM_POTENTIAL_SITES":
            line = i + 3
            for pot in xrange(numPotSites):
                # check that the list is in the correct order. die if not.
                if pot+1 != int(sdat[line+pot][0]):
                    sys.exit("Potential site list out of order!")
                potSites[pot][:] = sdat[line+pot][2:5]
    return potSites

### bondAnalysis.boo file functions.

# Lets define some functions which extract the relevent information from the
# bondAnalysis.boo file.

def BooBoo(baboo):
    """
    This function returns the bond orientational order for the atoms contained
    in a 2D array created by reading the "bondAnalysis.boo" file.
    """
    # number of atoms is in the first line.
    numAtoms = int(baboo[0][0])
    boo = np.zeros(shape=(numAtoms))
    for atom in range(numAtoms):
        boo[atom] = baboo[atom+1][1] 
    return boo

### bondAnalysis.bl file functions.

# Lets define some functions which extract the relevent information from the
# bondAnalysis.bl file.

def BlNumAtoms(babl):
    """
    This function returns the number of atoms in a system, extracted from a 2D
    array created from reading the bondanalysis.bl file.
    """
    i = 1
    for j in range(len(babl)):
        if babl[-(i)][1] == "Num_bonds:":
            vals = re.split('_',babl[-i][0])
            return int(vals[1])
        else:
            i += 1

def BlNumBonds(babl):
    """
    This function returns the number of bonds per atom contained in a 2D array
    created by reading the bondAnalysis.bl file. If this array is not passed,
    it is contructed here.
    """
    numAtoms = BlNumAtoms(babl)
    numBonds = np.zeros(shape=(numAtoms), dtype = int)
    atom = 0
    for item in range(len(babl)): # loop over the first index of the array.
        if babl[item][1] == "Num_bonds:": 
            numBonds[atom] = int(babl[item][2])
            atom += 1
    return numBonds

def BlBondingArray(babl):
    """
    This function returns a list, containg the atoms to which a certain atom is
    bonded. This information is contained a 2D array passed to this function, 
    which is created by reading the bondAnalysis.bl file. 

    IMPORTANT NOTE: This currently works with only > 4 bonds per atom, OR the new
    bondAnalysis script that Naseer will work on. the new bondAnalysis script will
    output all the bonds in the same single line, instead of the current method of
    4 bonds per line.
    """
    numBondsList = BlNumBonds(babl)
    numAtoms     = BlNumAtoms(babl)
    i = 1
    bondedList = np.zeros(shape=(numAtoms), dtype = object)
    for atom in range(numAtoms):
        atomBonds = np.zeros(shape= (numBondsList[atom]), dtype = int)
        for bond in range(numBondsList[atom]):
            if i > len(babl):
                break
            l = bond * 2
            vals = re.split('_',babl[i][l])
            atomBonds[bond] = vals[1]
        bondedList[atom] = atomBonds
        i += 2
    return bondedList

def BlBondLengths(babl):
    """
    This function returns a list, containg magnitude of the bonds of an atom.
    This list is created from a passed array contructed from reading the 
    bondAnalysis.bl file. If the array is not passed, it is contructed here.
    """
    numBondsList = BlNumBonds(babl)
    numAtoms     = BlNumAtoms(babl)
    i = 1
    bondLengths = np.zeros(shape=(numAtoms), dtype = object)
    for atom in range(numAtoms):
        atomBonds = np.zeros(shape= (numBondsList[atom]))
        for bond in range(numBondsList[atom]):
            if i > len(babl):
                break
            l = (bond * 2) + 1
            atomBonds[bond] = babl[i][l]
        bondLengths[atom] = atomBonds
        i += 2
    return bondLengths

### bondAnalysis.ba file functions.

# Lets define functions which extract information from the bondAnalysis.ba 
# file.

def BaNumAtoms(baba):
    """
    This function returns the number of atoms from a 2D array created from reading
    the bondAnalysis.ba file.
    """
    numAtoms = 0
    for line in range(len(baba)):
        if baba[line][0] == 'Num':
            numAtoms += 1
    return int(numAtoms)

def BaNumAngles(baba):
    """
    This function returns the number of bond angles for an atom, extracted from
    a 2D array which is created from reading the bondAnalysis.ba file.
    """
    numAtoms = BaNumAtoms(baba)
    bondAngles = np.zeros(shape=(numAtoms), dtype = int)
    atom = 0
    for i in range(len(baba)):
        if baba[i][0] == "Num":
            bondAngles[atom] = (int(baba[i][3]))
            atom += 1
    return bondAngles

def BaBondAngleList(baba):
    """
    This function returns the angles (in degrees) for an atom. This angle
    is the angle made between two bonds of the atom in question. For exmple,
    the bond angle for O in H2O is ~ 104.5 degrees.
    """
    numAtoms = BaNumAtoms(baba)
    bondAngleList = np.zeros(shape=(numAtoms), dtype = object)
    nAngles = BaNumAngles(baba)
    i = 1
    for atom in range(numAtoms):
        angleList = np.zeros(shape=(nAngles[atom]))
        for angle in range(nAngles[atom]):
            angleList[angle] = baba[i][6]
            i += 1
        bondAngleList[atom] = angleList
        i += 1
    return bondAngleList

