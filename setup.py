import setuptools

setuptools.setup(
    name='ISMgas',
    version='1.0',
    author='Keerthi Vasan G C',
    author_email='kvgc@ucdavis.edu',
    description='ISM gas kinematics code',
    # long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/kvgc/ISMgas',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.10',
        'scipy>=1.1',
        'astropy>=3.1',
        'matplotlib>=3.0'
    ],
)
