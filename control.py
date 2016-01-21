import numpy as np
import olcaoPy.constants as co
import olcaoPy.fileOps as fo
import sys
import os
import re
import math
import copy
import random
import collections
import json

class Structure(object):

    def __init__(self, fileName = None, buf = 10.0):
        """
        This object takes a file with the handle "fileName", reads it, and
        breaks it up into useful information and stores them into an object.
        This object has the following fields:

        :title      The title of the structure
        :cellInfo   The a, b, c vectors and the alpha, beta, and gamma angles
                    associated with this structure.
        :coordType  The way we notate the atomic coordinates. Currently this
                    is "F" for fractional, and "C" for cartesian.
        :numAtoms   The total number of atoms in the system.
        :atomCoors  The coordiantes of the atoms in the system.
        :atomNames  The "names" of the atoms in the system. Note that the
                    name is simply the string representing the name, and
                    that is not necessarily the element. For instance,
                    in the olcao.skl file you may have "si4_2" for the
                    name of the element. here, si is the element, 4 is
                    the specie number, and the 2 is the type number.
                    There are methods supplied to extract each one of
                    these in the structure object.
        :spaceGrp   The space group number for this structure.
        :supercell  The supercell size in a, b, and c directions.
        :cellType   The cell type. either "F" for full, or "P" for
                    primitive.
        :rlm        real lattice matrix. this is the projection of
                    a, b, and c vectors on the x, y, and z axis. it is
                    of the form: ax ay az
                                 bx by bz
                                 cx cy cz
        :mlr        the inverse of the real lattice matrix. the name is
                    a silly joke.

        If the fileName is not passed to the contructor, then an empty
        structure object is made.

        The "buf" parameter is used when the structure is contructed from
        an ".xyz" object.


        """
        if fileName != None:
            # get the extention.
            extention = os.path.splitext(fileName)[1]
            # load information from file based on the extention, or throw
            # an error if the extention isnt unrecognized.

            if extention == ".skl":
                # read the file into a 2D array of strings.
                skl = fo.readFile(fileName)

                # get the information out of it.
                self.title     = fo.SklTitle(skl)
                self.cellInfo  = fo.SklCellInfo(skl)
                self.coordType = fo.SklCoordType(skl)
                self.numAtoms  = fo.SklNumAtoms(skl)
                self.atomCoors = fo.SklCoors(skl)
                self.atomNames = fo.SklAtomNames(skl)
                self.spaceGrp  = fo.SklSpaceGrp(skl)
                self.supercell = fo.SklSupercell(skl)
                self.cellType  = fo.SklCellType(skl)

                # calculate the real lattice matrix and its inverse
                self.rlm = self.makeRealLattice()
                self.mlr = self.makeRealLatticeInv()
            elif extention == ".xyz":
                # read the file into a 2D string array.
                xyz = fo.readFile(fileName)

                # get the information.
                self.title = fo.XyzComment(xyz)
                self.coordType = "C" # always cartesian in an xyz file
                self.numAtoms = fo.XyzNumAtoms(xyz)
                self.atomCoors = fo.XyzCoors(xyz)
                self.atomNames = fo.XyzAtomNames(xyz)
                self.spaceGrp = "1_a" # no symmetry info in xyz files.
                self.supercell = np.ones(3) # no super cell.
                self.cellType = "F" # always "full" type in xyz files.

                self.cellInfo = self.computeCellInfo(buf)
                self.rlm = self.makeRealLattice()
                self.mlr = self.makeRealLatticeInv()
                self.shiftXyzCenter(buf)
            elif extention == ".dat":
                sdat = fo.readFile(fileName)
                self.title = "Generic structure.dat title"
                self.coordType = "C"
                self.numAtoms = fo.SdatNumAtomSites(sdat)
                self.atomCoors = fo.SdatAtomSites(sdat)
                self.atomNames = fo.SdatAtomNames(sdat)
                self.spaceGrp = "1_a"
                self.supercell = np.ones(3)
                self.cellType = "F"
                # the next 3 fields get reset when we call the "toSI" method,
                # but ill take the small performance hit for transparancy's
                # sake.
                self.rlm = fo.SdatCellVecs(sdat)
                self.mlr = np.linalg.inv(self.rlm)
                self.cellInfo = self.getCellInfoFromRlm()
                self.toSI()
            else:
                sys.exit("Unknown file extention: " + str(extention))
        else:
            self.title     = None
            self.cellInfo  = None
            self.coordType = None
            self.numAtoms  = None
            self.atomCoors = None
            self.atomNames = None
            self.spaceGrp  = None
            self.superCell = None
            self.cellType  = None
            self.rlm       = None
            self.mlr       = None

    def makeRealLattice(self):
        """
        This function creats the real lattice matrix from the magnitudes of the
        a, b, and c vectors contained in a typical olcao.skl file.
        """
        # lets redefine the passed parameters using the physical names. This is
        # wasteful, but makes the code more readable.
        a = self.cellInfo[0]
        b = self.cellInfo[1]
        c = self.cellInfo[2]
        # Convert the angles to radians.
        alpha = math.radians(self.cellInfo[3])
        beta  = math.radians(self.cellInfo[4])
        gamma = math.radians(self.cellInfo[5])
        # Start the construction of the RLM, the real lattice array.
        rlm = np.zeros(shape=(3,3), dtype=float)
        # Assume that a and x are coaxial.
        rlm[0][0] = a
        rlm[0][1] = 0.0
        rlm[0][2] = 0.0
        # b is then in the xy-plane.
        rlm[1][0] = (b * math.cos(gamma))
        rlm[1][1] = (b * math.sin(gamma))
        rlm[1][2] = 0.0
        # c is a mix of x,y, and z directions.
        rlm[2][0] = (c * math.cos(beta))
        rlm[2][1] = (c * (math.cos(alpha) - math.cos(gamma)*math.cos(beta)) /
                math.sin(gamma))
        rlm[2][2] = (c * math.sqrt(1.0 - math.cos(beta)**2 -
                ((rlm[2][1]/c)**2)))
        # now lets correct for numerical errors.
        rlm[rlm < 0.000000001] = 0.0
        return rlm

    def makeRealLatticeInv(self):
        """
        This function inverts the real lattice matrix, which is created from the
        magnitude of the cell vectors and the angles of a structure.
        """
        invRlm = np.linalg.inv(self.rlm)
        return invRlm

    def computeCellInfo(self, buf = 10.0):
        '''
        This function computs the a, b, and c lattice vectors for a set
        of coordinates passed to it. It is assumed that the system is in
        a orthorhombic box and therefore alpha, beta, and gamma are set
        to 90 degrees, and a, b, and c lattice vectors map to x, y, and z
        axes.
        '''
        cellInfo = np.zeros(6)
        cellInfo[0] = (self.atomCoors.max(axis=0)[0] -
                self.atomCoors.min(axis=0)[0] + buf)
        cellInfo[1] = (self.atomCoors.max(axis=0)[1] -
                self.atomCoors.min(axis=0)[1] + buf)
        cellInfo[2] = (self.atomCoors.max(axis=0)[2] -
                self.atomCoors.min(axis=0)[2] + buf)
        cellInfo[3] = 90.0
        cellInfo[4] = 90.0
        cellInfo[5] = 90.0
        return cellInfo

    def shiftXyzCenter(self, buf = 10.0):
        '''
        This function will linearly transelate all the atom along the
        orthogonal axes, to make sure that the system as a whole is
        centered in the simulation box. The atoms in the system will
        have a buffer from the edges of the simulation box, and
        therefore this is only applicable to molecular systems or
        clusters. Do not use this subroutine directly unless you know what
        you are doing.
        '''
        self.toCart()
        for i in self.atomCoors:
            i[:] += (buf/2.0)
        return self

    def clone(self):
        """
        This function clones a structure. in other words, this will generate
        a deep copy of the structure, such that modifying the clone does
        not effect the original in anyway.
        """
        clone = Structure()
        clone.title = self.title
        clone.cellInfo = np.copy(self.cellInfo)
        clone.coordType = self.coordType
        clone.numAtoms = self.numAtoms
        clone.atomCoors = np.copy(self.atomCoors)
        clone.atomNames = copy.deepcopy(self.atomNames)
        clone.spaceGrp = self.spaceGrp
        clone.supercell = np.copy(self.supercell)
        clone.cellType = self.cellType
        clone.rlm = np.copy(self.rlm)
        clone.mlr = np.copy(self.mlr)
        return clone

    def writeSkl(self, fileName   = "olcao.skl"):
        """
        This method will write a structure to a olcao.skl file. the
        default file name is "olcao.skl" if it is not passed. This method
        will fail if the file already exists, such that it does not
        accidentally overwrite files.
        """
        if os.path.isfile(fileName):
            sys.exit("File " + fileName + " already exists!")

        # concatenate the infomation into a string for printing.
        string  = "title\n"
        string += self.title
        string += "\nend\ncell\n"
        string += (" ".join(str(x) for x in self.cellInfo))
        string += "\n"
        if self.coordType == "F":
            string += "frac "
        elif self.coordType == "C":
            string += "cart "
        else:
            sys.exit("Unknown coordinate type " + self.coordType)
        string = string + str(self.numAtoms) + "\n"
        for i in xrange(self.numAtoms):
            string += str(self.atomNames[i])
            string += " "
            string += (" ".join(str(x) for x in self.atomCoors[i]))
            string += "\n"
        string += "space "
        string += self.spaceGrp
        string += "\n"
        string += "supercell "
        string += (" ".join(str(x) for x in self.supercell))
        string += "\n"
        if self.cellType == "F":
            string += "full"
        elif self.cellType == "P":
            string += "prim"
        else:
            sys.exit("Unknow cell type: " + str(self.cellType))

        with open(fileName, "w") as f:
            f.write(string)

        return self

    def writeXyz(self, comment = None, fileName = "out.xyz" ):
        """
        This function writes the structure in an xyz format. This format
        starts with the number of atoms, followed by a comment, and then
        numAtoms lines containing the element name and the cartesian
        coordinates for each atom.

        this function has to optional parameters, the comment (without
        the new line), and the file name to which we write.
        """
        # make sure we are not overwriting a file that already exists.
        if os.path.isfile(fileName):
            sys.exit("File " + fileName + " already exists!")

        if comment == None:
            comment = "Auto-generated comment"
        self.toCart()
        string = ""
        string += str(self.numAtoms)
        string += "\n"
        string += comment
        elementalNames = self.elementNames()
        for i in xrange(self.numAtoms):
            string += "\n"
            string += (elementalNames[i][:1].upper() + elementalNames[i][1:])
            string += " "
            string += (" ".join(str(x) for x in self.atomCoors[i]))

        string += "\n"
        with open(fileName, 'w') as f:
            f.write(string)

        return self

    def mutate(self, mag, prob):
        '''
        mutate moves the atoms in a structure by a given distance, mag, in
        a random direction. the probablity that any given atom will be moves
        is given by the argument 'prob', where 0.0 means %0 chance that  the
        atom will not move, and 1.0 means a %100 chance that a given atom will
        move.
        '''
        self.toCart()
        for i in xrange(self.numAtoms):
            if random.random() < prob:
                theta = random.random() * math.pi
                phi = random.random() * 2.0 * math.pi
                x = mag * math.sin(theta) * math.cos(phi)
                y = mag * math.sin(theta) * math.sin(phi)
                z = mag * math.cos(theta)
                self.atomCoors[i] += [x, y, z]
        self.applyPBC()
        self.spaceGrp = "1_a"
        return self

    def elementList(self):
        """
        This subroutine returns a list of unique elements in a
        given structure. by "element", we mean the elements on the
        priodic table, such as "H", "Li", "Ag", etc.
        """
        elementList = []
        for atom in xrange(self.numAtoms):
            # the atom name may contain numbers indicating the type or
            # species of the elements such as "y4" or "ca3_1". so we will
            # split the name based on the numbers, taking only the first
            # part which contains the elemental name.
            name = re.split(r'(\d+)', self.atomNames[atom])[0]
            if name not in elementList:
                elementList.append(name)
        return elementList

    def speciesList(self):
        """
        This subroutine return a list of unique species in a
        given structure. by 'species', we mean the following: in a
        typical olcao.skl file, the atom's names maybe the (lower case)
        atomic name, plus some numbers, such as 'si3' or 'o2'. however
        this number may be missing in which case it is assumed to be the
        default value of '1'. at any rate, the number defines the
        'specie' of this atom, thus allowing us to treat the different
        species differently. for instance, in Si2N4, the two Si are in
        very different environments. therefore, we can treat them
        differently by designating them as si1 and si2.
        """
        speciesList = []
        for atom in xrange(self.numAtoms):
            if self.atomNames[atom] not in speciesList:
                speciesList.append(self.atomNames[atom])
        return speciesList

    def elementNames(self):
        """
        This subroutine returns the atom element names of the atoms in the
        system. the atom element name is the name of the element of the
        atom without any added digits or other marks.
        """
        atomElementList = []
        for atom in xrange(self.numAtoms):
            name = re.split(r'(\d+)', self.atomNames[atom])[0]
            atomElementList.append(name)
        return atomElementList

    def atomZNums(self):
        """
        This subroutine returns the Z number of the atoms in the
        system.
        """
        atomElementNames = self.elementNames()
        atomZNums        = []
        for atom in xrange(self.numAtoms):
            atomZNums.append(co.atomicZ[atomElementNames[atom]])
        return atomZNums

    def toFrac(self):
        """
        Modifies the structure, by converting its atomic coordinates to
        fractional. If the coordinates are already fractional, nothing
        is done.
        """
        if self.coordType == 'F':
            return
        for i in self.atomCoors:
            (i[0], i[1], i[2])=(sum(i[:] * self.mlr[:][0]),
                                sum(i[:] * self.mlr[:][1]),
                                sum(i[:] * self.mlr[:][2]))

        self.coordType = 'F'
        return self

    def toCart(self):
        """
        Modifies the structure, by converting its atomic coordinates to
        cartesian. If the coordinates are already cartesian, nothing
        is done.
        """
        if self.coordType == 'C':
            return
        for i in self.atomCoors:
            (i[0], i[1], i[2])=(sum(i[:] * self.rlm[:][0]),
                                sum(i[:] * self.rlm[:][1]),
                                sum(i[:] * self.rlm[:][2]))

        self.coordType = 'C'
        return self

    def minDistMat(self):
        """
        This function creates the min. distance matrix between all the points
        in the system that is passed to this function.
        The min. distance here is the distance between two points when the
        periodic boundary conditions (PBCs) are taken into account. For example,
        consider the (silly) 1D system:

            [ABCDEFG]

        where the distance between A and G is normally calculated across the
        BCDEF atoms. However, when PBCs are considered they are immediately
        connected since:

            ...ABCDEFGABCDEFG....
               ^     ^^     ^
        and so on.
        """
        mdm = np.zeros(shape=(self.numAtoms, self.numAtoms))
        mdm[:][:] = 1000000000.0

        atom = np.zeros(3)
        self.toCart()
        for a in xrange(self.numAtoms):
            for x in xrange(-1, 2):
                for y in xrange(-1, 2):
                    for z in xrange(-1, 2):
                        atom = np.copy(self.atomCoors[a])
                        # convert the copy to fractional, move it, and convert
                        # back to cartesian. then calculate distance from all
                        # other atoms in the system.
                        (atom[0],atom[1],atom[2])=(sum(atom[:]*self.mlr[:][0]),
                                                   sum(atom[:]*self.mlr[:][1]),
                                                   sum(atom[:]*self.mlr[:][2]))
                        atom[:] += [float(x), float(y), float(z)]
                        (atom[0],atom[1],atom[2])=(sum(atom[:]*self.rlm[:][0]),
                                                   sum(atom[:]*self.rlm[:][1]),
                                                   sum(atom[:]*self.rlm[:][2]))
                        for b in xrange(a+1, self.numAtoms):
                            dist = math.sqrt(sum((atom-self.atomCoors[b])*
                                                 (atom-self.atomCoors[b])))
                            if dist < mdm[a][b]:
                                mdm[a][b] = dist
                                mdm[b][a] = dist

        for i in xrange(self.numAtoms):
            mdm[i][i] = 0.0

        return mdm

    def minDistVecs(self):
        '''
        This function creates the min. distance matrix between all the points
        in the system that is passed to this function.
        The min. distance here is the distance between two points when the
        periodic boundary conditions (PBCs) are taken into account. For example,
        consider the (silly) 1D system:

            [ABCDEFG]

        where the distance between A and G is normally calculated across the
        BCDEF atoms. However, when PBCs are considered they are immediately
        connected since:

            ...ABCDEFGABCDEFG....
               ^     ^^     ^
        and so on.

        furthermore, this function returns the min. distance vectors. this
        vector points from the an atom in the original "box", to the nearest
        version of all other atoms, when PBCs are considered.
        '''
        mdm = np.zeros(shape=(self.numAtoms, self.numAtoms))
        mdm[:][:] = 1000000000.0
        mdv = np.zeros(shape=(self.numAtoms, self.numAtoms, 3))

        atom = np.zeros(3)
        self.toCart()
        for a in xrange(self.numAtoms):
            for x in xrange(-1, 2):
                for y in xrange(-1, 2):
                    for z in xrange(-1, 2):
                        atom = np.copy(self.atomCoors[a])
                        # convert the copy to fractional, move it, and convert
                        # back to cartesian. then calculate distance from all
                        # other atoms in the system.
                        (atom[0],atom[1],atom[2])=(sum(atom[:]*self.mlr[:][0]),
                                                   sum(atom[:]*self.mlr[:][1]),
                                                   sum(atom[:]*self.mlr[:][2]))
                        atom[:] += [float(x), float(y), float(z)]
                        (atom[0],atom[1],atom[2])=(sum(atom[:]*self.rlm[:][0]),
                                                   sum(atom[:]*self.rlm[:][1]),
                                                   sum(atom[:]*self.rlm[:][2]))
                        for b in xrange(a+1, self.numAtoms):
                            dist = math.sqrt(sum((atom-self.atomCoors[b])*
                                                 (atom-self.atomCoors[b])))
                            if dist < mdm[a][b]:
                                mdm[a][b] = dist
                                mdm[b][a] = dist

                                mdv[a][b][:] = self.atomCoors[b] - atom
                                mdv[b][a][:] = atom - self.atomCoors[b]
        for i in xrange(self.numAtoms):
            mdv[i][i][:] = 0.0

        return mdv

    def applyPBC(self):
        '''
        This function ensures that all atoms are within a box that is defined
        by the a, b, and c lattice vectors. In other words, this function
        ensures that all atoms in the system are placed between 0 and 1 in
        fractional coordinates.
        '''
        self.toFrac()
        for a in range(self.numAtoms):
            while self.atomCoors[a][0] > 1.0:
                self.atomCoors[a][0] -= 1.0
            while self.atomCoors[a][1] > 1.0:
                self.atomCoors[a][1] -= 1.0
            while self.atomCoors[a][2] > 1.0:
                self.atomCoors[a][2] -= 1.0
            while self.atomCoors[a][0] < 0.0:
                self.atomCoors[a][0] += 1.0
            while self.atomCoors[a][1] < 0.0:
                self.atomCoors[a][1] += 1.0
            while self.atomCoors[a][2] < 0.0:
                self.atomCoors[a][2] += 1.0

        return self

    def coFn(self, dist, cutoffRad):
        '''
        coFn returns the value of the cutoff function as defined in:

        "Atom Centered Symmetry Fucntions for Contructing High-Dimentional
        Neural Network Potentials",
        by Jorg Behler, J. Chem. Phys. 134, 074106 (2011)
        '''
        if dist > cutoffRad:
            return 0.0
        return (0.5 * (math.cos((math.pi*dist)/cutoffRad) + 1.0))

    def genSymFn1(self, cutoff, mdm = None):
        '''
        This function generates the first symmetry function G1 as defined in:

        "Atom Centered Symmetry Fucntions for Contructing High-Dimentional
        Neural Network Potentials",
        by Jorg Behler, J. Chem. Phys. 134, 074106 (2011)
        '''
        if mdm == None:
            mdm = self.minDistMat()
        symFn1 = np.zeros(self.numAtoms)
        for i in range(self.numAtoms):
            for j in range(self.numAtoms):
                symFn1[i] += self.coFn(mdm[i][j], cutoff)

        return symFn1

    def genSymFn2(self, cutoff, rs, eta, mdm = None):
        '''
        This function generates the second symmetry function G2 as defined in:

        "Atom Centered Symmetry Fucntions for Contructing High-Dimentional
        Neural Network Potentials",
        by Jorg Behler, J. Chem. Phys. 134, 074106 (2011)
        '''
        if mdm == None:
            mdm = self.minDistMat()
        symFn2 = np.zeros(self.numAtoms)
        for i in range(self.numAtoms):
            for j in range(self.numAtoms):
                val = self.coFn(mdm[i][j], cutoff)
                if val != 0.0:
                    symFn2[i] += (math.exp(-eta*((mdm[i][j]-rs)**2)) * val)

        return symFn2

    def genSymFn3(self, cutoff, kappa, mdm = None):
        '''
        This function generates the third symmetry function G3 as defined in:

        "Atom Centered Symmetry Fucntions for Contructing High-Dimentional
        Neural Network Potentials",
        by Jorg Behler, J. Chem. Phys. 134, 074106 (2011)
        '''
        if mdm == None:
            mdm = self.minDistMat()
        symFn3 = np.zeros(self.numAtoms)
        for i in range(self.numAtoms):
            for j in range(self.numAtoms):
                val = self.coFn(mdm[i][j], cutoff)
                if val != 0.0:
                    symFn3[i] += (math.cos(kappa * mdm[i][j]) * val)

        return symFn3

    def genSymFn4(self, cutoff, lamb, zeta, eta, mdm = None, mdv = None):
        '''
        This function generates the forth symmetry function G4 as defined in:

        "Atom Centered Symmetry Fucntions for Contructing High-Dimentional
        Neural Network Potentials",
        by Jorg Behler, J. Chem. Phys. 134, 074106 (2011)
        '''
        if mdm == None:
            mdm = self.minDistMat()
        if mdv == None:
            mdv = self.minDistVecs()
        symFn4 = np.zeros(self.numAtoms)

        # vectors from atom i to j, and from i to k.
        Rij = np.zeros(3)
        Rik = np.zeros(3)

        for i in range(self.numAtoms):
            for j in range(self.numAtoms):
                if i == j:
                    continue
                FcRij = self.coFn(mdm[i][j], cutoff)
                if FcRij == 0.0:
                    continue
                Rij = mdv[i][j][:]
                for k in range(self.numAtoms):
                    if i == k or j == k:
                        continue
                    FcRik = self.coFn(mdm[i][k], cutoff)
                    if FcRik == 0.0:
                        continue
                    FcRjk = self.coFn(mdm[j][k], cutoff)
                    if FcRjk == 0.0:
                        continue
                    Rik = mdv[i][k][:]
                    # the cos of the angle is the dot product devided
                    # by the multiplication of the magnitudes. the
                    # magnitudes are the distances in the mdm.
                    cosTheta_ijk = np.dot(Rij, Rik) / (mdm[i][j] * mdm[i][k])
                    symFn4[i] += (((1.0+lamb*cosTheta_ijk)**zeta) *
                                  (math.exp(-eta*(mdm[i][j]**2 + mdm[i][k]**2 +
                                      mdm[j][k]**2))) * FcRij * FcRik * FcRjk)

            symFn4[i] *= (2**(1.0 - zeta))

        return symFn4


    def genSymFn5(self, cutoff, lamb, zeta, eta, mdm = None, mdv = None):
        '''
        This function generates the fifth symmetry function G5 as defined in:

        "Atom Centered Symmetry Fucntions for Contructing High-Dimentional
        Neural Network Potentials",
        by Jorg Behler, J. Chem. Phys. 134, 074106 (2011)
        '''
        if mdm == None:
            mdm = self.minDistMat()
        if mdv == None:
            mdv = self.minDistVecs()
        symFn5 = np.zeros(self.numAtoms)

        # vectors from atom i to j, and from i to k.
        Rij = np.zeros(3)
        Rik = np.zeros(3)

        for i in range(self.numAtoms):
            for j in range(self.numAtoms):
                if i == j:
                    continue
                FcRij = self.coFn(mdm[i][j], cutoff)
                if FcRij == 0.0:
                    continue
                Rij = mdv[i][j]
                for k in range(self.numAtoms):
                    if i == k:
                        continue
                    FcRik = self.coFn(mdm[i][k], cutoff)
                    if FcRik == 0.0:
                        continue
                    Rik = mdv[i][k]
                    # the cos of the angle is the dot product devided
                    # by the multiplication of the magnitudes. the
                    # magnitudes are the distances in the mdm.
                    cosTheta_ijk = np.dot(Rij, Rik) / (mdm[i][j] * mdm[i][k])
                    symFn5[i] += (((1.0+lamb*cosTheta_ijk)**zeta) *
                                  (math.exp(-eta*(mdm[i][j]**2 +
                                             mdm[i][k]**2)))*
                                  (FcRij * FcRik))

            symFn5[i] *= (2**(1.0 - zeta))

        return symFn5

    def getSymFns(self):
        '''
        This function returns a set of 79 symmetry functions that describe
        the local atomic environment for each atom in a structure.
        The set was chosen after some experimentation. So while the set is
        good, it might not be the best, or the most efficient way to
        describe the local env. of atoms.
        '''
        allSym = np.zeros(shape=(self.numAtoms, 79))
        mdm = self.minDistMat()
        mdv = self.minDistVecs()
        print "Working on the symmetry function 1 set..."
        allSym[:, 0]  = self.genSymFn1(1.0, mdm)
        allSym[:, 1]  = self.genSymFn1(1.2, mdm)
        allSym[:, 2]  = self.genSymFn1(1.4, mdm)
        allSym[:, 3]  = self.genSymFn1(1.6, mdm)
        allSym[:, 4]  = self.genSymFn1(1.8, mdm)
        allSym[:, 5]  = self.genSymFn1(2.0, mdm)
        allSym[:, 6]  = self.genSymFn1(2.2, mdm)
        allSym[:, 7]  = self.genSymFn1(2.4, mdm)
        allSym[:, 8]  = self.genSymFn1(2.6, mdm)
        allSym[:, 9]  = self.genSymFn1(2.8, mdm)
        allSym[:, 10] = self.genSymFn1(3.0, mdm)
        allSym[:, 11] = self.genSymFn1(3.2, mdm)
        allSym[:, 12] = self.genSymFn1(3.4, mdm)
        allSym[:, 13] = self.genSymFn1(3.6, mdm)
        allSym[:, 14] = self.genSymFn1(3.8, mdm)
        allSym[:, 15] = self.genSymFn1(4.0, mdm)
        allSym[:, 16] = self.genSymFn1(4.2, mdm)
        allSym[:, 17] = self.genSymFn1(4.4, mdm)
        allSym[:, 18] = self.genSymFn1(4.6, mdm)
        allSym[:, 19] = self.genSymFn1(4.8, mdm)
        allSym[:, 20] = self.genSymFn1(5.0, mdm)
        allSym[:, 21] = self.genSymFn1(5.2, mdm)
        allSym[:, 22] = self.genSymFn1(5.4, mdm)
        allSym[:, 23] = self.genSymFn1(5.6, mdm)
        allSym[:, 24] = self.genSymFn1(5.8, mdm)
        allSym[:, 25] = self.genSymFn1(6.0, mdm)

        print "Working on the symmetry function 2 set..."
        allSym[:, 26] = self.genSymFn2(5.0, 2.0, 12.0, mdm)
        allSym[:, 27] = self.genSymFn2(5.0, 2.2, 12.0, mdm)
        allSym[:, 28] = self.genSymFn2(5.0, 2.4, 12.0, mdm)
        allSym[:, 29] = self.genSymFn2(5.0, 2.6, 12.0, mdm)
        allSym[:, 30] = self.genSymFn2(5.0, 2.8, 12.0, mdm)
        allSym[:, 31] = self.genSymFn2(5.0, 3.0, 12.0, mdm)
        allSym[:, 32] = self.genSymFn2(5.0, 3.2, 12.0, mdm)
        allSym[:, 33] = self.genSymFn2(5.0, 3.4, 12.0, mdm)
        allSym[:, 34] = self.genSymFn2(5.0, 3.6, 12.0, mdm)
        allSym[:, 35] = self.genSymFn2(5.0, 3.8, 12.0, mdm)
        allSym[:, 36] = self.genSymFn2(5.0, 4.0, 12.0, mdm)
        allSym[:, 37] = self.genSymFn2(5.0, 4.2, 12.0, mdm)
        allSym[:, 38] = self.genSymFn2(5.0, 4.4, 12.0, mdm)
        allSym[:, 39] = self.genSymFn2(5.0, 4.6, 12.0, mdm)
        allSym[:, 40] = self.genSymFn2(5.0, 4.8, 12.0, mdm)

        print "Working on the symmetry function 3 set..."
        allSym[:, 41] = self.genSymFn3(5.0, 1.0, mdm)
        allSym[:, 42] = self.genSymFn3(5.0, 1.5, mdm)
        allSym[:, 43] = self.genSymFn3(5.0, 2.0, mdm)
        allSym[:, 44] = self.genSymFn3(5.0, 2.5, mdm)
        allSym[:, 45] = self.genSymFn3(5.0, 3.0, mdm)
        allSym[:, 46] = self.genSymFn3(5.0, 3.5, mdm)
        allSym[:, 47] = self.genSymFn3(5.0, 4.0, mdm)
        allSym[:, 48] = self.genSymFn3(5.0, 4.5, mdm)

        print "Working on the symmetry function 4 set..."
        allSym[:, 49] = self.genSymFn4(5.0, 1.0, 1.0, 0.05, mdm, mdv)
        allSym[:, 50] = self.genSymFn4(5.0, 1.0, 2.0, 0.05, mdm, mdv)
        allSym[:, 51] = self.genSymFn4(5.0, 1.0, 3.0, 0.05, mdm, mdv)
        allSym[:, 52] = self.genSymFn4(5.0, 1.0, 4.0, 0.05, mdm, mdv)
        allSym[:, 53] = self.genSymFn4(5.0, 1.0, 5.0, 0.05, mdm, mdv)
        allSym[:, 54] = self.genSymFn4(5.0, 1.0, 1.0, 0.07, mdm, mdv)
        allSym[:, 55] = self.genSymFn4(5.0, 1.0, 2.0, 0.07, mdm, mdv)
        allSym[:, 56] = self.genSymFn4(5.0, 1.0, 3.0, 0.07, mdm, mdv)
        allSym[:, 57] = self.genSymFn4(5.0, 1.0, 4.0, 0.07, mdm, mdv)
        allSym[:, 58] = self.genSymFn4(5.0, 1.0, 5.0, 0.07, mdm, mdv)
        allSym[:, 59] = self.genSymFn4(5.0, 1.0, 1.0, 0.09, mdm, mdv)
        allSym[:, 60] = self.genSymFn4(5.0, 1.0, 2.0, 0.09, mdm, mdv)
        allSym[:, 61] = self.genSymFn4(5.0, 1.0, 3.0, 0.09, mdm, mdv)
        allSym[:, 62] = self.genSymFn4(5.0, 1.0, 4.0, 0.09, mdm, mdv)
        allSym[:, 63] = self.genSymFn4(5.0, 1.0, 5.0, 0.09, mdm, mdv)

        print "Working on the symmetry function 5 set..."
        allSym[:, 64] = self.genSymFn5(5.0, 1.0, 1.0, 0.3, mdm, mdv)
        allSym[:, 65] = self.genSymFn5(5.0, 1.0, 2.0, 0.3, mdm, mdv)
        allSym[:, 66] = self.genSymFn5(5.0, 1.0, 3.0, 0.3, mdm, mdv)
        allSym[:, 67] = self.genSymFn5(5.0, 1.0, 4.0, 0.3, mdm, mdv)
        allSym[:, 68] = self.genSymFn5(5.0, 1.0, 5.0, 0.3, mdm, mdv)
        allSym[:, 69] = self.genSymFn5(5.0, 1.0, 1.0, 0.4, mdm, mdv)
        allSym[:, 70] = self.genSymFn5(5.0, 1.0, 2.0, 0.4, mdm, mdv)
        allSym[:, 71] = self.genSymFn5(5.0, 1.0, 3.0, 0.4, mdm, mdv)
        allSym[:, 72] = self.genSymFn5(5.0, 1.0, 4.0, 0.4, mdm, mdv)
        allSym[:, 73] = self.genSymFn5(5.0, 1.0, 5.0, 0.4, mdm, mdv)
        allSym[:, 74] = self.genSymFn5(5.0, 1.0, 1.0, 0.5, mdm, mdv)
        allSym[:, 75] = self.genSymFn5(5.0, 1.0, 2.0, 0.5, mdm, mdv)
        allSym[:, 76] = self.genSymFn5(5.0, 1.0, 3.0, 0.5, mdm, mdv)
        allSym[:, 77] = self.genSymFn5(5.0, 1.0, 4.0, 0.5, mdm, mdv)
        allSym[:, 78] = self.genSymFn5(5.0, 1.0, 5.0, 0.5, mdm, mdv)

        return allSym

    def covalentRadii(self):
        '''
        This function returns a dictionary where the keys are the unique
        elements in the system and the values are their covalent radii.
        '''
        eList = self.elementList()
        covRads = {}
        for element in eList:
            covRads[element] = co.covalRad[element]
        return covRads

    def bondingList(self,
                    mdm = None,
                    bf = 1.1,
                    covRads = None,
                    atomElementList = None):
        '''
        This function creates a list of dicts. for every atom x, we get a dict
        of atoms to which x is bonded.
        The optional arguments are "covRads", which is a list of the covalent
        radii for every unique element, and "mdm" which is the min. distance
        between atoms, when PBCs are taken into consideration, and "bf" that is
        is the bonding factor. atoms are considered to be bonded if their
        distance is less or equal to the sum of their covalent radii * bf.
        The default for bf is 1.1.
        NOTE: The returned list starts at 0. so atom 102 will be returned
        as atom 101, etc.
        '''
        if mdm == None:
            mdm = self.minDistMat()
        if covRads == None:
            covRads = self.covalentRadii()
        if atomElementList == None:
            atomElementList = self.elementNames()

        bondingList = collections.defaultdict(dict)
        for atom1 in xrange(self.numAtoms-1):
            for atom2 in xrange(atom1 + 1, self.numAtoms):
                bondDist = (covRads[atomElementList[atom1]] +
                            covRads[atomElementList[atom2]]) * bf
                if bondDist >= mdm[atom1][atom2]:
                    bondingList[atom1][atom2] = mdm[atom1][atom2]
                    bondingList[atom2][atom1] = mdm[atom2][atom1]

        return bondingList

    def coordination(self, bondingList = None, bf = 1.1):
        '''
        This function returns the coordination of each atom in the structure.
        by coordination of an atom x, we mean the number of atoms to which x is
        bonded.
        '''
        if bondingList == None:
            bondingList = self.bondingList(bf = bf)
        return [len(bondingList[i]) for i in xrange(len(bondingList))]

    def toAU(self):
        pass

    def toSI(self):
        '''
        This function takes a structure object and converts it to SI units.
        '''
        self.rlm *= co.AU2SI
        self.cellInfo = self.getCellInfoFromRlm()
        self.mlr = np.linalg.inv(self.rlm)
        self.atomCoors *= co.AU2SI
        return self

    def getCellInfoFromRlm(self):
        '''
        This function computes the magnitudes of the a, b, and c cell
        vectors as well as the alpha, beta, gamma angles. together, these
        six parameters are the cell information.
        '''
        cellInfo = np.zeros(6)
        for i in xrange(3):
            cellInfo[i] = math.sqrt(self.rlm[i][0]*self.rlm[i][0] +
                                    self.rlm[i][1]*self.rlm[i][1] +
                                    self.rlm[i][2]*self.rlm[i][2])

        # alpha is angle between b and c
        cellInfo[3] = math.acos(np.dot(self.rlm[1], self.rlm[2])/
            (np.linalg.norm(self.rlm[1]) * np.linalg.norm(self.rlm[2])))
        cellInfo[3] = math.degrees(cellInfo[3])

        # beta is between a and c
        cellInfo[4] = math.acos(np.dot(self.rlm[0], self.rlm[2])/
            (np.linalg.norm(self.rlm[0]) * np.linalg.norm(self.rlm[2])))
        cellInfo[4] = math.degrees(cellInfo[4])

        # gamma is between a and b
        cellInfo[5] = math.acos(np.dot(self.rlm[0], self.rlm[1])/
            (np.linalg.norm(self.rlm[0]) * np.linalg.norm(self.rlm[1])))
        cellInfo[5] = math.degrees(cellInfo[5])
        return cellInfo

    def writeLAMMPS(self, fileName = "data.lmp"):
        '''
        This function prints a file, "data.lmp" by default, that can
        be used as input to LAMMPS. this input is used by lammps to
        determine the location and the types of the atoms in the
        system.
        '''
        # make sure we are not overwriting a file already made.
        if os.path.isfile(fileName):
            sys.exit("File " + fileName + " already exists!")

        # convert to cartesian. lammps input must be in cartesian.
        self.toCart()

        string = ''
        string += "AutoGenerated title from conversionTools\n"
        string += str(self.numAtoms)
        string += " atoms\n"
        numTypes = self.elementList()
        string += str(len(numTypes))
        string += " atom types\n"
        string += "0.0 "
        string += str(self.cellInfo[0])
        string += " xlo xhi\n"
        string += "0.0 "
        string += str(self.cellInfo[1])
        string += " ylo yhi\n"
        string += "0.0 "
        string += str(self.cellInfo[2])
        string += " zlo zhi\n"
        string += "\nAtoms\n\n"

        # instead of element names, such as 'si' or 'ca', lammps uses
        # type IDs which are integers starting from 1 up to the total
        # number of types in the system. so we need to make a dictionary
        # to map element names to types.
        # note that the notion of 'type' in lammps is distinct from the
        # one used in olcao. there is no direct correlation.
        counter = 1
        nameDict = {}
        for i in range(len(numTypes)):
            nameDict[numTypes[i]] = counter
            counter += 1
        # print the mapping between element names and type IDs
        with open(fileName, 'w') as f:
            f.write(json.dumps(nameDict))

        # lammps atom numbers start from 1 to numatoms.
        counter = 1
        for i in range(self.numAtoms):
            string += str(i+1)
            string += " "
            string += str(nameDict[self.atomNames[i]])
            string += " "
            string += (" ".join(str(x) for x in self.atomCoors[i]))
            string += "\n"

        with open(fileName, 'w') as f:
            f.write(string)
        return self
