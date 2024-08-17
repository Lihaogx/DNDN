# DNDN

Implementation for the paper "Dynamic Neural Dowker Networks: Approximating Persistent Homology in Dynamic Graphs."

## Requirements

The necessary packages are listed in requirements.txt and can be installed using the following command:

```
pip install -r requirements.txt
```

## Setup Cython
'sg2dgm' is derived from the paper "Neural Approximation of Graph Topological Features"(https://arxiv.org/pdf/2201.12032.pdf), which provides a library for computing the persistence image.

```
cd ./sg2dgm
python setup_PI.py build_ext --inplace
```

to setup ./sg2dgm/persistenceImager.pyx

If the command does not work, a substitute solution is to copy the code in ./sg2dgm/persistenceImager.pyx to a new file named ./sg2dgm/persistenceImager.py, this might also work.


## Run example experiments for DNDN

```
python main.py --config './config/config.yaml'
```
You can modify the necessary configurations in 'config.yml'.

## Custom Datasets

'utils.process_data' is the method we provide for processing your custom datasets. Its main functionalities include generating dynamic Dowker filtration ground truths and constructing dynamic graph classification datasets, etc. You can refer to the function 'construct_sp_dataset()' for constructing custom datasets.

## Data
Some data has been uploaded here: https://drive.google.com/drive/folders/1LpBPT6PG4RyOSkr_MgxWuIYn3iI1RO4Q?usp=drive_link"