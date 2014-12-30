from distutils.core import setup

name = 'pyPaSWAS'
setup(
    #Information about the package
    name=name,
    version='0.1.0',
    description='Python implementation of Smith-Waterman on CUDA',
    author='Sven Warris',
    author_email='s.warris@pl.hanze.nl',
    url='http://trac.nbic.nl/' + name.lower(),
    license='',  #TODO Provide a license

    # We probably need more requirements, such as pycuda
    requires=['numpy', 'Bio (>1.54)', 'pycuda (>=2012.1)'],

    # What we'll package as part of this dist
    packages=[name, name + '.Core',
              name + '.Core.cfg',
              name + '.Core.test'],
     )
