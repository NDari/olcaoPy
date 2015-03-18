## Executable utility scripts ##

These are the scripts that allows a user to do some of the more common tasks typically
needed while doing reasearch in condensed matter physics in general, and the OLCAO
package in prticular. What follows is a short dircription of the scripts, and some
of their options and use cases. For the most up to date version of the options for
each script, run:

scriptname -h

# conversionTools #

This script is used to:

* Convert between Cartesian and fractional coordinates.
* Convert between various common file formats, such as .skl or .xyz
* add padding to .xyz structures, such the molecule/system is away from the edges of the
    simulation box.
*

# genSym #

This script generate a set of 79 symmetry functions, which are chosen to decribe the
local environments of pure elemental atoms. This means that they are not very useful
for that purpose in multi-elemental systems as they do not (yet) differentiate between
the different elements.

for a complete discussion of symmetry functions, see:

"Atom Centered Symmetry Fucntions for Contructing High-Dimentional Neural Network 
Potentials", by Jorg Behler, J. Chem. Phys. 134, 074106 (2011)


This software is Free. Do with it what you like. I am not responsible for any bugs!

2013-2015 Nasser A. Dari, Computational Physics Group, UMKC.
