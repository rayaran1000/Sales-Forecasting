#Libraries imported to use find_packages, setup and List functionality
from setuptools import find_packages,setup
from typing import List

#Use of Hyhen_e_dot explained later.
HYPHEN_E_DOT='-e .'
#Function to convert requirements.txt to list format
#Then returning the list of libraries in the requirements file 
#below in setup para to install the packages.
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]  
#If we get the "-e ." in requirments.txt, no need to treat it as a package, so 
#removing it from the requirements file.
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

#Creating the package for our ML model and installing the packages present in
# the requirements file.
setup(
    name='SalesForecasting',
    version='0.0.1',
    author='Aranya Ray',
    author_email='aranya.ray1998@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)