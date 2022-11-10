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
* Python 3.7-3.9 (https://www.python.org/downloads/) or anaconda (https://docs.anaconda.com/anaconda/install/index.html)
* Tensorflow (https://www.tensorflow.org/install)
* Tensorflow_addons library (https://www.tensorflow.org/addons/overview)
* For graphic neural network, the stellargraph graph library is needed. 

  pip install stellargraph or conda install -c stellargraph stellargraph

* IDAES

  See https://github.com/IDAES/idaes-pse and https://idaes-pse.readthedocs.io/en/stable/tutorials/getting_started/index.html for more details.

## Collaborators
Collaborators: [PNNL](https://www.pnnl.gov/), [NETL](https://www.netl.doe.gov/), and [UW](https://www.washington.edu/) 

<img src="./docs/images/Pacific_Northwest_National_Laboratory_logo.svg.png" alt="PNNL-logo" height="60" img align="left"> <img src="./docs/images/NETL.png" alt="NETL-logo" height="60" img align="center"> <img src="./docs/images/UW.png" alt="UW-logo" height="45" img align="center"> 
<br/><br/>

## Funding
Funding: [ARPA-E Differentiate](https://arpa-e.energy.gov/technologies/programs/differentiate)

<img src="./docs/images/ARPA-E_logo_2021.png" alt="ARPAE-logo" height="60" img align="center"> <img src="./docs/images/Differentiate.png" alt="differentiate-logo" height="60" img align="center">  
<br/><br/>
