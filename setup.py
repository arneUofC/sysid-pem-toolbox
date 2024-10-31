from setuptools import setup, find_packages

setup(
    name="sysid_pem_toolbox",
    version="0.1.0",
    description="A System Identification and PEM Toolbox",
    author="Arne Dankers",
    author_email="arne.dankers2@ucalgary.ca",
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
