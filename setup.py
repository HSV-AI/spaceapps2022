from setuptools import setup, find_packages

import pathlib
import pkg_resources
import setuptools


# # NOTE bad practice
# with pathlib.Path('src/requirements.txt').open() as requirements_txt:
#     install_requires = [
#         str(requirement)
#         for requirement
#         in pkg_resources.parse_requirements(requirements_txt)
#     ]

setup(
    name='SpaceApps2022',
    version='0.0.1',
    url='https://github.com/HSV-AI/spaceapps2022.git',
    author='HSV-AI',
    author_email='author@gmail.com',
    description='Description of my package',
    packages=find_packages(where='src'),    
    # install_requires=install_requires,
)