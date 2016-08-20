import sys
import os
import re
import math
import copy
import random
import collections
import json
import itertools
import numpy as np
import constants as co

class Structure(object):

    def __init__(self, file_name=None, buf=10.0):
        """
        This object takes a file with the handle "file_name", reads it, and
        breaks it up into useful information and stores them into an object.
        This object has the following fields:

        :title
                    The title of the structure
        :cell_info
                    The a, b, c vectors and the alpha, beta, and gamma angles
                    associated with this structure.
        :coordinate_type
                    The way we notate the atomic coordinates. Currently this
                    is "F" for fractional, and "C" for Cartesian.
        :num_atoms
                    The total number of atoms in the system.
        :atom_coordinates
                    The coordinates of the atoms in the system.
        :atom_names
                    The "names" of the atoms in the system. Note that the
                    name is simply the string representing the name, and
                    that is not necessarily the element. For instance,
                    in the olcao.skl file you may have "si4_2" for the
                    name of the element. here, si is the element, 4 is
                    the specie number, and the 2 is the type number.
                    There are methods supplied to extract each one of
                    these in the structure object.
        :space_group
                    The space group number for this structure.
        :supercell
                    The supercell size in a, b, and c directions.
        :supercell_mirror
                    Used sometimes for things......
        :cell_type
                    The cell type. Either "F" for full, or "P" for
                    primitive.
        :real_lattice_matrix
                    real lattice matrix. This is the projection of
                    a, b, and c vectors on the x, y, and z axis. it is
                    of the form: ax ay az
                                 bx by bz
                                 cx cy cz
        :recip_lattice_matrix
                    the inverse of the real lattice matrix. The name is
                    a silly joke.

        If the file_name is not passed to the constructor, then an empty
        structure object is made.

        The "buf" parameter is used when the structure is constructed from
        an ".xyz" object, and it denotes the distance between the outer-most
        atoms from the edges of the box containing the structure.


        """
        if file_name is not None:
            # get the extension.
            extension = os.path.splitext(file_name)[1]
            # load information from file based on the extension, or throw
            # an error if the extension is not unrecognized.

            if extension == ".skl":
                # read the file into a 2D array of strings.
                skl = []
                with open(file_name, 'r') as file:
                    lines = [line.strip() for line in file.readlines()]
                    for line in lines:
                        skl.append(re.split(" +", line))

                # get the information out of it.
                line = 1  # skip 0th line with "title"
                self.title = ''
                while skl[line][0] != 'end':
                    self.title += " ".join(skl[line])
                    self.title += "\n"
                    line += 1
                self.title = self.title.rstrip('\n')
                line += 2  # skip the line with "end" and "cell"
                self.cell_info = [float(x) for x in skl[line]]
                line += 1
                if 'frac' in skl[line][0]:
                    self.coordinate_type = 'F'
                elif 'cart' in skl[line][0]:
                    self.coordinate_type = 'C'
                else:
                    sys.exit("Unknown coordinate type: " + skl[line][0])
                self.num_atoms = int(skl[line][1])
                line += 1
                self.atom_coordinates = np.zeros((self.num_atoms, 3))
                self.atom_names = []
                for i in range(self.num_atoms):
                    self.atom_names.append(skl[line+i][0].lower())
                    self.atom_coordinates[i] = skl[line+i][1:4]
                line += self.num_atoms
                self.space_group = skl[-3][1]  # space group is 3rd from bottom.
                line += 1
                self.supercell = np.zeros(shape=(3), dtype=int)
                self.supercell[0:3] = skl[-2][1:4]
                self.supercell_mirror = np.zeros(shape=(3), dtype=int)
                self.supercell_mirror = skl[-2][4:7]
                if skl[-1][0] == "full":
                    self.cell_type = "F"
                elif skl[-1][0] == "prim":
                    self.cell_type = "P"
                else:
                    sys.exit("Unknown cell type: " + skl[-1][0])

                # calculate the real lattice matrix and its inverse
                self.real_lattice_matrix = Structure.make_real_lattice(self.cell_info)
                self.recip_lattice_matrix = Structure.make_recip_lattice_matrix(self.cell_info)
            # elif extension == ".xyz":
            #     # read the file into a 2D string array.
            #     xyz = fo.readFile(file_name)
            #
            #     # get the information.
            #     self.title = fo.XyzComment(xyz)
            #     self.coordinate_type = "C"  # always Cartesian in an xyz file
            #     self.num_atoms = fo.XyzNumAtoms(xyz)
            #     self.atom_coordinates = fo.XyzCoors(xyz)
            #     self.atom_names = fo.XyzAtomNames(xyz)
            #     self.space_group = "1_a"  # no symmetry info in xyz files.
            #     self.supercell = [1, 1, 1]  # no super cell.
            #     self.cell_type = "F"  # always "full" type in xyz files.
            #
            #     self.cell_info = self.compute_cell_info(buf)
            #     self.real_lattice_matrix = self.make_real_lattice()
            #     self.recip_lattice_matrix = self.make_recip_lattice_matrix()
            #     self.shift_xyz_center(buf)
            # elif extension == ".dat":
            #     sdat = fo.readFile(file_name)
            #     self.title = "Generic structure.dat title"
            #     self.coordinate_type = "C"
            #     self.num_atoms = fo.SdatNumAtomSites(sdat)
            #     self.atom_coordinates = fo.SdatAtomSites(sdat)
            #     self.atom_names = fo.SdatAtomNames(sdat)
            #     self.space_group = "1_a"
            #     self.supercell = np.ones(3)
            #     self.cell_type = "F"
            #     # the next 3 fields get reset when we call the "toSI" method,
            #     # but ill take the small performance hit for transparancy's
            #     # sake.
            #     self.real_lattice_matrix = fo.SdatCellVecs(sdat)
            #     self.recip_lattice_matrix = np.linalg.inv(self.real_lattice_matrix)
            #     self.cell_info = self.cell_info_from_real_lattice_matrix()
            #     self.toSI()
            else:
                sys.exit("Unknown file extension: " + str(extension))
        else:  # file_name is None
            self.title = None
            self.cell_info = None
            self.coordinate_type = None
            self.num_atoms = None
            self.atom_coordinates = None
            self.atom_names = None
            self.space_group = None
            self.super_cell = None
            self.cell_type = None
            self.real_lattice_matrix = None
            self.recip_lattice_matrix = None

    @staticmethod
    def make_real_lattice(cell_info):
        """
        This function creats the real lattice matrix from the magnitudes of the
        a, b, and c vectors contained in a typical olcao.skl file.
        """
        # lets redefine the passed parameters using the physical names. This is
        # wasteful, but makes the code more readable.
        a_vector = cell_info[0]
        b_vector = cell_info[1]
        c_vector = cell_info[2]
        # Convert the angles to radians.
        alpha = math.radians(cell_info[3])
        beta = math.radians(cell_info[4])
        gamma = math.radians(cell_info[5])
        # Start the construction of the RLM, the real lattice array.
        real_lattice_matrix = np.zeros(shape=(3, 3), dtype=float)
        # Assume that a and x are coaxial.
        real_lattice_matrix[0:3] = [a_vector, 0.0, 0.0]
        # b is then in the xy-plane.
        real_lattice_matrix[1][0] = (b_vector * math.cos(gamma))
        real_lattice_matrix[1][1] = (b_vector * math.sin(gamma))
        real_lattice_matrix[1][2] = 0.0
        # c is a mix of x,y, and z directions.
        real_lattice_matrix[2][0] = (c_vector * math.cos(beta))
        real_lattice_matrix[2][1] = (c_vector *
                                     (math.cos(alpha) - math.cos(gamma)*math.cos(beta)) /
                                     math.sin(gamma))
        real_lattice_matrix[2][2] = (c_vector *
                                     math.sqrt(1.0 - math.cos(beta)**2 -
                                               ((real_lattice_matrix[2][1]/c_vector)**2)))
        # now lets correct for numerical errors.
        real_lattice_matrix[real_lattice_matrix < 0.000000001] = 0.0
        return real_lattice_matrix

    @staticmethod
    def make_recip_lattice_matrix(cell_info):
        """
        This function inverts the real lattice matrix, which is created from
        the magnitude of the cell vectors and the angles of a structure.
        """
        # lets redefine the passed parameters using the physical names. This is
        # wasteful, but makes the code more readable.
        a_vector = cell_info[0]
        b_vector = cell_info[1]
        c_vector = cell_info[2]
        # Convert the angles to radians.
        alpha = math.radians(cell_info[3])
        beta = math.radians(cell_info[4])
        gamma = math.radians(cell_info[5])
        # Start the construction of the RLM, the real lattice array.
        recip_lattice_matrix = np.zeros(shape=(3, 3), dtype=float)
        v = math.sqrt(1.0 - (math.cos(alpha) * math.cos(alpha)) -
                      (math.cos(beta) * math.cos(beta)) -
                      (math.cos(gamma) * math.cos(gamma)) +
                      (2.0 * math.cos(alpha) * math.cos(beta) * math.cos(gamma)))

        # assume a and x are coaxial.
        recip_lattice_matrix[0][0] = 1.0 / a_vector
        recip_lattice_matrix[0][1] = 0.0
        recip_lattice_matrix[0][2] = 0.0

        # then b is in the xy-plane.
        recip_lattice_matrix[1][0] = -math.cos(gamma) / (a_vector * math.sin(gamma))
        recip_lattice_matrix[1][1] = 1.0 / (b_vector * math.sin(gamma))
        recip_lattice_matrix[1][2] = 0.0

        # c is then a mix of all three axes.
        recip_lattice_matrix[2][0] = ((math.cos(alpha) * math.cos(gamma) - math.cos(beta)) /
                                      (a_vector * v * math.sin(gamma)))
        recip_lattice_matrix[2][1] = ((math.cos(beta)*math.cos(gamma) - math.cos(alpha)) /
                                      (b_vector * v * math.sin(gamma)))
        recip_lattice_matrix[2][2] = (math.sin(gamma)) / (c_vector * v)

        # now lets correct for numerical errors.
        recip_lattice_matrix[recip_lattice_matrix < 0.000000001] = 0.0
        return recip_lattice_matrix

    def to_frac(self):
        """
        Modifies the structure, by converting its atomic coordinates to
        fractional. If the coordinates are already fractional, nothing
        is done.
        """
        if self.coordinate_type == 'F':  # nothing to do.
            return self
        self.atom_coordinates = self.atom_coordinates.dot(self.recip_lattice_matrix)
        self.coordinate_type = 'F'
        return self

    def to_cart(self):
        """
        Modifies the structure, by converting its atomic coordinates to
        Cartesian. If the coordinates are already Cartesian, nothing
        is done.
        """
        if self.coordinate_type == 'C':  # nothing to do.
            return self
        self.atom_coordinates = self.atom_coordinates.dot(self.real_lattice_matrix)
        self.coordinate_type = 'C'
        return self

    def compute_cell_info(self, buf=10.0):
        '''
        This function computs the a, b, and c lattice vectors for a set
        of coordinates passed to it. It is assumed that the system is in
        a orthorhombic box and therefore alpha, beta, and gamma are set
        to 90 degrees, and a, b, and c lattice vectors map to x, y, and z
        axes.
        '''
        cell_info = np.zeros(6)
        cell_info[0] = (self.atom_coordinates.max(axis=0)[0] -
                        self.atom_coordinates.min(axis=0)[0] + buf)
        cell_info[1] = (self.atom_coordinates.max(axis=0)[1] -
                        self.atom_coordinates.min(axis=0)[1] + buf)
        cell_info[2] = (self.atom_coordinates.max(axis=0)[2] -
                        self.atom_coordinates.min(axis=0)[2] + buf)
        cell_info[3] = 90.0
        cell_info[4] = 90.0
        cell_info[5] = 90.0
        return cell_info

    def shift_xyz_center(self, buf=10.0):
        """
        This function will linearly transelate all the atom along the
        orthogonal axes, to make sure that the system as a whole is
        centered in the simulation box. The atoms in the system will
        have a buffer from the edges of the simulation box, and
        therefore this is only applicable to molecular systems or
        clusters. Do not use this subroutine directly unless you know what
        you are doing.
        """
        if self.coordinate_type == "F":
            self.to_cart()
            self.atom_coordinates += (buf/2.0)
            self.to_frac()
        else:
            self.atom_coordinates += (buf/2.0)
        return self

    def clone(self):
        """
        This function clones a structure. in other words, this will generate
        a deep copy of the structure, such that modifying the clone does
        not effect the original in anyway.
        """
        clone = Structure()
        clone.title = self.title
        clone.cell_info = np.copy(self.cell_info)
        clone.coordinate_type = self.coordinate_type
        clone.num_atoms = self.num_atoms
        clone.atom_coordinates = np.copy(self.atom_coordinates)
        clone.atom_names = copy.deepcopy(self.atom_names)
        clone.space_group = self.space_group
        clone.supercell = np.copy(self.supercell)
        clone.cell_type = self.cell_type
        clone.real_lattice_matrix = np.copy(self.real_lattice_matrix)
        clone.recip_lattice_matrix = np.copy(self.recip_lattice_matrix)
        return clone

    def write_skl(self, file_name="olcao.skl"):
        """
        This method will write a structure to a olcao.skl file. the
        default file name is "olcao.skl" if it is not passed. This method
        will fail if the file already exists, such that it does not
        accidentally overwrite files.
        """
        if os.path.isfile(file_name):
            sys.exit("File " + file_name + " already exists!")

        # concatenate the infomation into a string for printing.
        string = "title\n"
        string += self.title
        string += "\nend\ncell\n"
        string += (" ".join(str(x) for x in self.cell_info))
        string += "\n"
        if self.coordinate_type == "F":
            string += "frac "
        elif self.coordinate_type == "C":
            string += "cart "
        else:
            sys.exit("Unknown coordinate type " + self.coordinate_type)
        string = string + str(self.num_atoms) + "\n"
        for i in range(self.num_atoms):
            string += self.atom_names[i]
            string += " "
            string += (" ".join(str(x) for x in self.atom_coordinates[i]))
            string += "\n"
        string += "space "
        string += self.space_group
        string += "\n"
        string += "supercell "
        string += (" ".join(str(x) for x in self.supercell))
        string += "\n"
        if self.cell_type == "F":
            string += "full"
        elif self.cell_type == "P":
            string += "prim"
        else:
            sys.exit("Unknow cell type: " + str(self.cell_type))

        with open(file_name, "w") as file:
            file.write(string)

        return self

    def write_xyz(self, comment=None, file_name="out.xyz"):
        """
        This function writes the structure in an xyz format. This format
        starts with the number of atoms, followed by a comment, and then
        num_atoms lines containing the element name and the Cartesian
        coordinates for each atom.

        This function has to optional parameters, the comment (without
        the new line), and the file name to which we write.
        """
        # make sure we are not overwriting a file that already exists.
        if os.path.isfile(file_name):
            sys.exit("File " + file_name + " already exists!")

        if comment is None:
            comment = "Auto-generated comment"
        converted_to_cart = False
        if self.coordinate_type == "F":
            self.to_cart()
            converted_to_cart = True
        string = ""
        string += str(self.num_atoms)
        string += "\n"
        string += comment
        elements = self.element_names()
        for i in range(self.num_atoms):
            string += "\n"
            string += (elements[i][:1].upper() + elements[i][1:])
            string += " "
            string += (" ".join(str(x) for x in self.atom_coordinates[i]))

        string += "\n"
        with open(file_name, 'w') as file:
            file.write(string)

        if converted_to_cart:
            self.to_frac()
        return self

    def mutate(self, mag, prob):
        '''
        mutate moves the atoms in a structure by a given distance, mag, in
        a random direction. the probablity that any given atom will be moves
        is given by the argument 'prob', where 0.0 means %0 chance that  the
        atom will not move, and 1.0 means a %100 chance that a given atom will
        move.
        '''
        converted_to_cart = False
        if self.coordinate_type == "F":
            self.to_cart()
            converted_to_cart = True
        for i in range(self.num_atoms):
            if random.random() < prob:
                theta = random.random() * math.pi
                phi = random.random() * 2.0 * math.pi
                x_coords = mag * math.sin(theta) * math.cos(phi)
                y_coords = mag * math.sin(theta) * math.sin(phi)
                z_coords = mag * math.cos(theta)
                self.atom_coordinates[i] += [x_coords, y_coords, z_coords]
        self.apply_pbc()
        self.space_group = "1_a"
        if converted_to_cart:
            self.to_frac()
        return self

    def element_list(self):
        """
        This subroutine returns a list of unique elements in a
        given structure. By "element", we mean the elements on the
        priodic table, such as "H", "Li", "Ag", etc.
        """
        element_list = []
        for atom in range(self.num_atoms):
            # the atom name may contain numbers indicating the type or
            # species of the elements such as "y4" or "ca3_1". so we will
            # split the name based on the numbers, taking only the first
            # part which contains the elemental name.
            name = re.split(r'(\d+)', self.atom_names[atom])[0]
            if name not in element_list:
                element_list.append(name)
        return element_list

    def species_list(self):
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
        species_list = []
        for atom in range(self.num_atoms):
            if self.atom_names[atom] not in species_list:
                species_list.append(self.atom_names[atom])
        return species_list

    def element_names(self):
        """
        This subroutine returns the atom element names of the atoms in the
        system. the atom element name is the name of the element of the
        atom without any added digits or other marks.
        """
        atom_element_list = []
        for atom in range(self.num_atoms):
            name = re.split(r'(\d+)', self.atom_names[atom])[0]
            atom_element_list.append(name)
        return atom_element_list

    def atom_z_num(self):
        """
        This subroutine returns the Z number of the atoms in the
        system.
        """
        atom_element_name = self.element_names()
        atom_z_num = []
        for atom in range(self.num_atoms):
            atom_z_num.append(co.atomicZ[atom_element_name[atom]])
        return atom_z_num


    def min_dist_matrix(self):
        """
        This function creates the min. distance matrix between all the points
        in the system that is passed to this function.
        The min. distance here is the distance between two points when the
        periodic boundary conditions (PBCs) are taken into account. For
        example, consider the (silly) 1D system:

            [ABCDEFG]

        where the distance between A and G is normally calculated across the
        BCDEF atoms. However, when PBCs are considered they are immediately
        connected since:

            ...ABCDEFGABCDEFG....
               ^     ^^     ^
        and so on.
        """
        mdm = np.zeros(shape=(self.num_atoms, self.num_atoms))
        mdm = 1000000000.0

        atom = np.zeros(3)
        converted_to_cart = False
        if self.coordinate_type == "F":
            self.to_cart()
            converted_to_cart = True
        for atom in range(self.num_atoms):
            for x, y, z in itertools.product([-1, 0, 1], repeat=3):
                copy_atom = np.copy(self.atom_coordinates[atom])
                # convert the copy to fractional, move it, and convert
                # back to Cartesian. then calculate distance from all
                # other atoms in the system.
                copy_atom = copy_atom.dot(self.recip_lattice_matrix),
                copy_atom += [float(x), float(y), float(z)]
                copy_atom = copy_atom.dot(self.real_lattice_matrix)
                for mirror_atom in range(atom+1, self.num_atoms):
                    dist = math.sqrt(sum((copy_atom-self.atom_coordinates[mirror_atom]) *
                                         (copy_atom-self.atom_coordinates[mirror_atom])))
                    if dist < mdm[atom][mirror_atom]:
                        mdm[atom][mirror_atom] = dist
                        mdm[mirror_atom][atom] = dist

        for i in range(self.num_atoms):
            mdm[i][i] = 0.0

        if converted_to_cart:
            self.to_frac()
        return mdm

    def min_dist_vector(self, mdm=None):
        '''
        This function creates the min. distance matrix between all the points
        in the system that is passed to this function.
        The min. distance here is the distance between two points when the
        periodic boundary conditions (PBCs) are taken into account. For
        example, consider the (silly) 1D system:

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
        if mdm is None:
            mdm = self.min_dist_matrix()
        mdv = np.zeros(shape=(self.num_atoms, self.num_atoms, 3))

        atom = np.zeros(3)
        converted_to_cart = False
        if self.coordinate_type == "F":
            self.to_cart()
            converted_to_cart = True
        for atom in range(self.num_atoms):
            for x, y, z in itertools.product([-1, 0, 1], repeat=3):
                copy_atom = np.copy(self.atom_coordinates[atom])
                # convert the copy to fractional, move it, and convert
                # back to Cartesian. then calculate distance from all
                # other atoms in the system.
                copy_atom = atom.dot(self.recip_lattice_matrix),
                copy_atom += [float(x), float(y), float(z)]
                copy_atom = atom.dot(self.real_lattice_matrix)
                for mirror_atom in range(atom+1, self.num_atoms):
                    dist = math.sqrt(sum((atom-self.atom_coordinates[mirror_atom]) *
                                         (atom-self.atom_coordinates[mirror_atom])))
                    if dist < mdm[atom][mirror_atom]:
                        mdm[atom][mirror_atom] = dist
                        mdm[mirror_atom][atom] = dist
                        mdv[atom][mirror_atom] = self.atom_coordinates[mirror_atom] - copy_atom
                        mdv[mirror_atom][atom] = copy_atom - self.atom_coordinates[mirror_atom]

        for i in range(self.num_atoms):
            mdv[i][i] = 0.0

        if converted_to_cart:
            self.to_frac()
        return mdv

    def apply_pbc(self):
        '''
        This function ensures that all atoms are within a box that is defined
        by the a, b, and c lattice vectors. In other words, this function
        ensures that all atoms in the system are placed between 0 and 1 in
        fractional coordinates.
        '''
        converted_to_frac = False
        if self.coordinate_type == "C":
            self.to_frac()
            converted_to_frac = True
        for atom in range(self.num_atoms):
            while self.atom_coordinates[atom][0] > 1.0:
                self.atom_coordinates[atom][0] -= 1.0
            while self.atom_coordinates[atom][1] > 1.0:
                self.atom_coordinates[atom][1] -= 1.0
            while self.atom_coordinates[atom][2] > 1.0:
                self.atom_coordinates[atom][2] -= 1.0
            while self.atom_coordinates[atom][0] < 0.0:
                self.atom_coordinates[atom][0] += 1.0
            while self.atom_coordinates[atom][1] < 0.0:
                self.atom_coordinates[atom][1] += 1.0
            while self.atom_coordinates[atom][2] < 0.0:
                self.atom_coordinates[atom][2] += 1.0

        if converted_to_frac:
            self.to_cart()
        return self

    def covalent_radii(self):
        '''
        This function returns a dictionary where the keys are the unique
        elements in the system and the values are their covalent radii.
        '''
        elements = self.element_list()
        coval_radii = {}
        for element in elements:
            coval_radii[element] = co.covalRad[element]
        return coval_radii

    def bondingList(self, mdm=None, bonding_factor=1.1, coval_radii=None, elements=None):
        '''
        This function creates a list of dicts. for every atom x, we get a dict
        of atoms to which x is bonded.
        The optional arguments are "coval_radii", which is a list of the covalent
        radii for every unique element, and "mdm" which is the min. distance
        between atoms, when PBCs are taken into consideration.
        is the bonding factor. Atoms are considered to be bonded if their
        distance is less or equal to the sum of their covalent radii * bonding_factor.
        The default for bonding_factor is 1.1.
        NOTE: The returned list starts at 0. so atom 102 will be returned
        as atom 101, etc.
        '''
        if mdm is None:
            mdm = self.min_dist_matrix()
        if coval_radii is None:
            coval_radii = self.covalent_radii()
        if elements is None:
            elements = self.element_names()

        bondingList = collections.defaultdict(dict)
        for atom1 in range(self.num_atoms-1):
            for atom2 in range(atom1 + 1, self.num_atoms):
                bondDist = (coval_radii[elements[atom1]] +
                            coval_radii[elements[atom2]]) * bonding_factor
                if bondDist >= mdm[atom1][atom2]:
                    bondingList[atom1][atom2] = mdm[atom1][atom2]
                    bondingList[atom2][atom1] = mdm[atom2][atom1]

        return bondingList

    def coordination(self, bondingList=None, bonding_factor=1.1):
        '''
        This function returns the coordination of each atom in the structure.

        By coordination of an atom x, we mean the number of atoms to which x is
        bonded. Two atoms are considered bonded if the distance between them is
        less or equal to the sum of their covalent radii times a bonding factor
        which is passed to this function. The default value of the bonding
        factor is 1.1.
        '''
        if bondingList is None:
            bondingList = self.bondingList(bonding_factor=bonding_factor)
        return [len(bondingList[i]) for i in range(len(bondingList))]

    def toAU(self):
        pass

    def toSI(self):
        '''
        This function takes a structure object and converts it to SI units.
        '''
        self.real_lattice_matrix *= co.AU2SI
        self.cell_info = self.cell_info_from_real_lattice_matrix()
        self.recip_lattice_matrix = np.linalg.inv(self.real_lattice_matrix)
        self.atom_coordinates *= co.AU2SI
        return self

    def cell_info_from_real_lattice_matrix(self):
        '''
        This function computes the magnitudes of the a, b, and c cell
        vectors as well as the alpha, beta, gamma angles. Together, these
        six parameters are the cell information.
        '''
        cell_info = np.zeros(6)
        for i in range(3):
            cell_info[i] = math.sqrt(self.real_lattice_matrix[i][0]*self.real_lattice_matrix[i][0] +
                                     self.real_lattice_matrix[i][1]*self.real_lattice_matrix[i][1] +
                                     self.real_lattice_matrix[i][2]*self.real_lattice_matrix[i][2])

        # alpha is angle between b and c
        cell_info[3] = math.acos(np.dot(self.real_lattice_matrix[1], self.real_lattice_matrix[2]) /
                                 (np.linalg.norm(self.real_lattice_matrix[1]) * np.linalg.norm(self.real_lattice_matrix[2])))
        cell_info[3] = math.degrees(cell_info[3])

        # beta is between a and c
        cell_info[4] = math.acos(np.dot(self.real_lattice_matrix[0],
                                        self.real_lattice_matrix[2]) /
                                 (np.linalg.norm(self.real_lattice_matrix[0]) *
                                  np.linalg.norm(self.real_lattice_matrix[2])))
        cell_info[4] = math.degrees(cell_info[4])

        # gamma is between a and b
        cell_info[5] = math.acos(np.dot(self.real_lattice_matrix[0],
                                        self.real_lattice_matrix[1]) /
                                 (np.linalg.norm(self.real_lattice_matrix[0]) *
                                  np.linalg.norm(self.real_lattice_matrix[1])))
        cell_info[5] = math.degrees(cell_info[5])
        return cell_info

    def write_lammps(self, file_name="data.lmp"):
        '''
        This function prints a file, "data.lmp" by default, that can
        be used as input to LAMMPS. this input is used by lammps to
        determine the location and the types of the atoms in the
        system.
        '''
        # make sure we are not overwriting a file already made.
        if os.path.isfile(file_name):
            sys.exit("File " + file_name + " already exists!")

        # convert to Cartesian. lammps input must be in Cartesian.
        converted_to_cart = False
        if self.coordinate_type == "F":
            self.to_cart()
            converted_to_cart = True
        string = ''
        string += "Auto Generated title from conversionTools\n"
        string += str(self.num_atoms)
        string += " atoms\n"
        elements = self.element_list()
        string += str(len(elements))
        string += " atom types\n"
        string += "0.0 "
        string += str(self.cell_info[0])
        string += " xlo xhi\n"
        string += "0.0 "
        string += str(self.cell_info[1])
        string += " ylo yhi\n"
        string += "0.0 "
        string += str(self.cell_info[2])
        string += " zlo zhi\n"
        string += "\nAtoms\n\n"

        # instead of element names, such as 'si' or 'ca', lammps uses
        # type IDs which are integers starting from 1 up to the total
        # number of types in the system. So we need to make a dictionary
        # to map element names to types.
        # note that the notion of 'type' in lammps is distinct from the
        # one used in olcao. There is no direct correlation.
        counter = 1
        name_dict = {}
        for element in elements:
            name_dict[element] = counter
            counter += 1
        # print the mapping between element names and type IDs
        with open(file_name, 'w') as file:
            file.write(json.dumps(name_dict))

        # lammps atom numbers start from 1 to numatoms.
        counter = 1
        for i in range(self.num_atoms):
            string += str(i+1)
            string += " "
            string += str(name_dict[self.atom_names[i]])
            string += " "
            string += (" ".join(str(x) for x in self.atom_coordinates[i]))
            string += "\n"

        with open(file_name, 'w') as file:
            file.write(string)

        if converted_to_cart:
            self.to_frac()
        return self
