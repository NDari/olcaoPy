import unittest
import numpy as np
from control import Structure

class TestStructureMethods(unittest.TestCase):
    def test_init(self):
        s = Structure("test.skl")
        self.assertEqual(s.title,
            "Element Aluminium (Al)",
            "Title is incorrect")
        self.assertEqual(s.cellInfo[0], 10.0, "Cell a vector incorrect")
        self.assertEqual(s.cellInfo[1], 10.0, "Cell b vector incorrect")
        self.assertEqual(s.cellInfo[2], 10.0, "Cell c vector incorrect")

        self.assertEqual(s.coordType, "F", "Coordinate type incorrect")

        self.assertEqual(s.numAtoms, 1, "Number of atoms incorrect")

        self.assertEqual(list(s.rlm[0]), list([10.0, 0.0, 0.0]),
            "Real Lattice Matrix [0] is not correct")
        self.assertEqual(list(s.rlm[1]), list([0.0, 10.0, 0.0]),
            "Real Lattice Matrix [1] is not correct")
        self.assertEqual(list(s.rlm[2]), list([0.0, 0.0, 10.0]),
            "Real Lattice Matrix [2] is not correct")

        self.assertEqual(list(s.mlr[0]), list(np.linalg.inv(s.rlm)[0]),
            "Real Lattice Matrix Inverse [0] is not correct")
        self.assertEqual(list(s.mlr[1]), list(np.linalg.inv(s.rlm)[1]),
            "Real Lattice Matrix Inverse [1] is not correct")
        self.assertEqual(list(s.mlr[2]), list(np.linalg.inv(s.rlm)[2]),
            "Real Lattice Matrix Inverse [2] is not correct")

        self.assertEqual(s.spaceGrp, '225', "Space group is not correct")
        self.assertEqual(list(s.supercell), list([1, 1, 1]),
            "Supercell is not correct")
        self.assertEqual(s.cellType, 'F', "Cell type is not correct")

    def test_toCart(self):
        s = Structure("test.skl")
        s.toCart()
        self.assertEqual(s.coordType, "C", "Coordinate type incorrect")
        self.assertEqual(list(s.atomCoors[0]), list([5.0, 5.0, 5.0]))

    def test_toCart(self):
        s = Structure("test.skl")
        s.toCart().toFrac()
        self.assertEqual(s.coordType, "F", "Coordinate type incorrect")
        self.assertEqual(list(s.atomCoors[0]), list([0.5, 0.5, 0.5]))

if __name__ == '__main__':
    with open('test.skl', 'w') as f:
        string = ("title\n"
                  "Element Aluminium (Al)\n"
                  "end\n"
                  "cell\n"
                  "10.0 10.0 10.0  90.000  90.000 90.000\n"
                  "fractional 1\n"
                  "Al   0.5      0.5      0.5\n"
                  "space 225\n"
                  "supercell 1 1 1\n"
                  "full\n")
        f.write(string)
    unittest.main()
