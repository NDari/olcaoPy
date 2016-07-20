import unittest
import os
import filecmp
import numpy as np
from control import Structure

class TestStructureMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open('test.skl', 'w') as f:
            string = ("title\n"
                      "Fake structure\n"
                      "end\n"
                      "cell\n"
                      "10.0 10.0 10.0 90.0 90.0 90.0\n"
                      "frac 5\n"
                      "al 0.25 0.25 0.25\n"
                      "al 0.25 0.5 0.75\n"
                      "al 0.75 0.5 0.25\n"
                      "al 0.5 0.5 0.5\n"
                      "al 0.75 0.75 0.75\n"
                      "space 225\n"
                      "supercell 1 1 1\n"
                      "full")
            f.write(string)

    @classmethod
    def tearDownClass(cls):
        os.remove("test.skl")

    def test_init(self):
        s = Structure("test.skl")
        self.assertEqual(s.title, "Fake structure", "Title is incorrect")
        self.assertEqual(s.cell_info[0], 10.0, "Cell a vector incorrect")
        self.assertEqual(s.cell_info[1], 10.0, "Cell b vector incorrect")
        self.assertEqual(s.cell_info[2], 10.0, "Cell c vector incorrect")

        self.assertEqual(s.coordinate_type, "F", "Coordinate type incorrect")

        self.assertEqual(s.num_atoms, 5, "Number of atoms incorrect")

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

        self.assertEqual(s.space_group, '225', "Space group is not correct")
        self.assertEqual(list(s.supercell), list([1, 1, 1]),
            "Supercell is not correct")
        self.assertEqual(s.cell_type, 'F', "Cell type is not correct")

    def test_to_cart(self):
        s = Structure("test.skl")
        s.to_cart()
        self.assertEqual(s.coordinate_type, "C", "Coordinate type incorrect")
        self.assertEqual(list(s.atom_coordinates[0]), list([2.5, 2.5, 2.5]),
                "Converting to cartesian did not work correctly")

    def test_to_frac(self):
        s = Structure("test.skl")
        s.to_cart().to_frac()
        self.assertEqual(s.coordinate_type, "F", "Coordinate type incorrect")
        self.assertEqual(list(s.atom_coordinates[0]), list([0.25, 0.25, 0.25]),
                "Converting to fractional did not work correctly.")

    def test_compute_cell_info(self):
        s = Structure("test.skl")
        orig = np.copy(s.cell_info)
        s.compute_cell_info()  # uses default buf=10 which is what we need.
        self.assertEqual(list(orig), list(s.cell_info),
                "Computing the cell info did not return expected values.")

    def test_write_skl(self):
        s = Structure("test.skl")
        s.write_skl("olcao.skl")
        self.assertTrue(filecmp.cmp('test.skl', 'olcao.skl'),
                "Writing the .skl file did not match structure.")
        os.remove('olcao.skl')


if __name__ == '__main__':
    unittest.main()
