from setuptools import setup, find_packages

# Defines the package metadata and depencies

setup(
    name="sysid_pem_toolbox",
    version="0.1.0",
    author="Arne Dankers",
    author_email="arne.dankers2@ucalgary.ca",
    description="A System Identification and PEM Toolbox",
    long_description=open('README.md').read(),
    url='https://github.com/arneUofC/sysid-pem-toolbox',
    packages=find_packages(),
    install_requires=[
        "control",
        "numpy",  
        "scipy",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
