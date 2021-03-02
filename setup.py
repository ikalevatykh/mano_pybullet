"""A setuptools based setup module."""

from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='mano_pybullet',
    version='0.1.0',
    description='MANO-based hand models for the PyBullet simulator.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ikalevatykh/mano_pybullet',
    author='Igor Kalevatykh',
    author_email='kalevatykhia@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering',
    ],
    keywords='PyBullet MANO VR robotics',
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'chumpy',
        'transforms3d',
        'pybullet>=3.0.6',
    ],
    extras_require={
        'gym': ['gym>=0.17.3', 'imageio']
    }
)
