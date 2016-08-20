import unittest
import os
import olcaoPy.symfunc as symfunc
from olcaoPy.control import Structure

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

    def test_cutoff_function(self):
        val = symfunc.cutoff_function(10.0, 0.0)
        self.assertEqual(val, 0.0, "distannce greater than cutoff did not return 0")
        val = symfunc.cutoff_function(0.0, 10.0)
        self.assertEqual(val, 1.0, "distannce of 0.0 did not return 1.0")

if __name__ == '__main__':
    unittest.main()
