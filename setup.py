from setuptools import setup

setup(
    name='Kalkayotl',
    version='0.3.0',
    author='Javier Olivares',
    author_email='javier.olivares-romero@u-bordeaux.fr',
    packages=['Code'],
    url='http://perso.astrophy.u-bordeaux.fr/JOlivares/kalkayotl/index.html',
    license='COPYING',
    description='Star distance inference code',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Astronomy"
    ],
    python_requires='>=3.6',
    install_requires=[

    ],
    zip_safe=True
)