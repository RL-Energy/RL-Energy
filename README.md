# RL-Energy
Machine learning for process system design
## Table of Contents
* Introduction
* Installation
* Collaborators
* Funding
## Introduction
A reinforcement learning (RL) based automated system for chemical engineering systems conceptual design is introduced and demonstrated in this project. An artificial intelligence (AI) agent performs the conceptual design by automatically deciding which process-units are necessary for the desired system, picking the process-units from the candidate process-units pool (CPP), connecting them together, and optimizing the operation of the system for the user-defined system performance targets. The AI agent automatically selects units from the user-defined CPPs, connects them to construct flowsheets, and optimizes the system design according to the user’s desired objective. The AI agent automatically interacts with a physics-based system-level modeling and simulation toolset, the Institute for the Design of Advanced Energy System (IDAES) Integrated Platform, to guarantee the system design is physically consistent.

Three examples, hydrodealkylation of toluene (HDA), methanol synthesis systems and reduced order model (ROM) of solid oxide fuel cell multi-physics (SOFC-MP) model are provided to get started with the RL model. See Installation section and readme file in each example folder for more details.
## Installation

The following libraries are needed.
* Python

  For Windows system, download the installer package from https://www.python.org/downloads/windows/ versions 3.7-3.9 are recommended. For MAC OS user, it can be   download from https://www.python.org/downloads/macos/ 
Note that to run the examples of Jupyter Notebook, anaconda may be needed (https://docs.anaconda.com/anaconda/install/index.html)
The user can find the anaconda package for various operation systems on https://www.anaconda.com/products/distribution#Downloads 
* Tensorflow

  The RL code use TensorFlow as the backend. The user can download TensorFlow on the website (https://www.tensorflow.org/install). The RL code is based on TensorFlow 1.0, but TensorFlow 2.0 is compatible. TensorFlow can be installed by pip or docker. 

  Python:
  pip install tensorflow

  Docker:
  docker pull tensorflow/tensorflow:latest 
  docker run -it -p 8888:8888 tensorflow/tensorflow:latest-jupyter

  The GPU version is recommended if the user has Nvidia GPU support. Note that the cudatoolkit and cudnn libraries are needed when installing TensorFlow GPU version.
* IDAES (Institute for the Design of Advanced Energy Systems) library
  
  The IDAES library was integrated into the platform that supports the full process modeling from flowsheet design to dynamic optimization and control within a single modeling environment.

  Python:
  1.	Install IDAES.
  
      pip install idaes-pse
    
  2.	Run the idaes get-extensions command to install the compiled binaries. These binaries include solvers and function libraries.
  
      idaes get-extensions

  Anaconda:
  1.	Install IDAES.
  
      conda install --yes -c IDAES-PSE -c conda-forge idaes-pse
  
  2.	Run the idaes get-extensions command to install the compiled binaries. These binaries include solvers and function libraries.
  
      idaes get-extensions

  More details can be found on https://github.com/IDAES/idaes-pse and https://idaes-pse.readthedocs.io/en/stable/tutorials/getting_started/index.html.


* Tensorflow_addons library 
  
  Python:
  pip install tensorflow-addons

  Or download the library (https://www.tensorflow.org/addons/overview)
  
* Stellargraph graph library
  
  For graphic neural network, the stellargraph graph library is needed. 

  Python:
  pip install stellargraph

  Anaconda:
  conda install -c stellargraph stellargraph

  Or download the library (https://www.stellargraph.io/) 

## Collaborators
Collaborators: [PNNL](https://www.pnnl.gov/), [NETL](https://www.netl.doe.gov/), and [UW](https://www.washington.edu/) 

<img src="./docs/images/Pacific_Northwest_National_Laboratory_logo.svg.png" alt="PNNL-logo" height="60" img align="left"> <img src="./docs/images/NETL.png" alt="NETL-logo" height="60" img align="center"> <img src="./docs/images/UW.png" alt="UW-logo" height="45" img align="center"> 
<br/><br/>

## Funding
Funding: [ARPA-E Differentiate](https://arpa-e.energy.gov/technologies/programs/differentiate)

<img src="./docs/images/ARPA-E_logo_2021.png" alt="ARPAE-logo" height="60" img align="center"> <img src="./docs/images/Differentiate.png" alt="differentiate-logo" height="60" img align="center">  
<br/><br/>
