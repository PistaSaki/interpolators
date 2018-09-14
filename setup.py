from setuptools import setup, find_packages

setup(
    name = 'interpolators',
    version = '0.0.1',
    url = 'none',
    author = 'Author Name',
    author_email = 'author@gmail.com',
    description = 'Description of my package',
    packages = find_packages(),    
    install_requires = ["matplotlib", "toolz", "numpy", "piplot"],
)