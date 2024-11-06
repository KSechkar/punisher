[![DOI](https://zenodo.org/badge/807035579.svg)](https://doi.org/10.5281/zenodo.14046987)
# punisher
Python code accompanying the manuscript 'Model-guided gene circuit design for engineering genetically stable cell populations in diverse applications' (Sechkar and Steel 2024, Journal of the Royal Society Interface). The article proposes a novel biomolecular controller for countring mutation spread, called 'the Punisher', and investigates its performance using a resource-aware coarse-grained _E. coli_ cell model first published in [Sechkar et al. 2024](https://www.nature.com/articles/s41467-024-46410-9) 

## File organisation
The repository includes the following Python scripts:
- Jupyter notebooks _FigXX.ipynb_, found in folders with the same name - allow to reproduce the corresponding figures
- _cell_model.py_ - python implementation of the cell model simulator
- _synthetic_circuits.py_ - python script allowing to simulate different synthetic gene circuits hosted by the cell
- _get_steady_state.py_ - python script allowing to retrieve and analyse the steady stat of the cell hosting a given circuit
- _Fig2/design_guidance_tools.py_ - functions allowing to get analytical guidance for choosing the design parameters of the Punisher circuit. Used to create Figure 2 and Figure S6
- _Fig5/pop_simulator.py_ - python implementation of the simulator of a population of cells hosting the Punisher circuit and one additional synthetic burdensome gene
- _Fig5/Make Pop Model/XX.ipynb_ - Jupyter notebooks allowing to parameterise the cell population model described above using stochastic single-cell model simulations. Each notebook typically outputs the estimated mean switching time from one state of the Punisher circuit to another.
- _Fig5/Make Pop Model/switching_time_estimation_tools.ipynb_ - auxiliary functions required to run the switching time estimation notebooks described above
- _FigS7/varvol_find_scaling.ipynb_ - Jupyter notebook allowing to find a scaling factor for the synthetic burdensome gene's promoter strength to ensure consistency between the scenario considered in Figure S7 with the rest of the simulations. Namely, in Figure S7 the cell's volume changes over the course of the cell cycle and the burdensome gene is chromosomally integrated, being replicated at a user-defined cell cycle phase

## System requirements
The code was run and ensured to be functional with Python 3.12 on PCs running on Windows 10 Pro 22H2, Windows 11 Home 22H2 and Ubuntu 20.04.6. The software requirements can be found in the file _requirements.txt_

All scripts, except _FigS1.ipynb_, _FigS7.ipynb_, _FigS10.ipynb_, _varvol_find_scaling.ipynb_ and those in the _Make Pop Model_ folder, can be run on a normal PC CPU in under 30 minutes. We recommend that the longer-running scripts are run on a Linux PC with an NVIDIA GPU, which allows for the efficient parallel simulation of trajectories as part of the program.
