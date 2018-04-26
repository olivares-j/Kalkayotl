from distutils.core import setup

setup(
    name='Kalkayotl',
    version='0.1.0',
    author='Javier Olivares',
    author_email='javier.olivares-romero@u-bordeaux.fr',
    packages=['code'],
    url='http://perso.astrophy.u-bordeaux.fr/JOlivares/kalkayotl/index.html',
    license='COPYING',
    description='Simple parallax to distance converter',
    long_description=open('README.md').read(),
    install_requires=[
        "emcee = 2.2.1",
        "numpy >= 1.14",
        "scipy >= 1.0",
        "numpy >= 1.14",
        "matplotlib  >= 2.1.1",
        "pandas      >= 0.22",
        "progressbar >= 2.3",
        "astroML     >= 0.3"
    ],
)