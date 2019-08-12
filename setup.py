import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="labeledclusters",
    version="0.0.1",
    author="Andreas Hinterreiter",
    author_email="andreas.hinterreiter@jku.at",
    description="Python code for projecting, exporting and analyzing clusters of labeled, high-dimensional data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/einbandi/labeledclusters",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        
        "License :: OSI Approved :: MIT License",
        
        "Operating System :: OS Independent",
    ],
)