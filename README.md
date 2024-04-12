# DNDN

Implementation for the paper "Dynamic Neural Dowker Networks: Approximating Persistent Homology in Dynamic Graphs."

## Results

The experimental results in our paper are as follows:

### **Table.1 Datasets**

| Datasets  | Classes | Graphs (Small) | Avg. Nodes (Small) | Avg. Edges (Small) | Graphs (Large) | Avg. Nodes (Large) | Avg. Edges (Large) |
|-----------|---------|----------------|--------------------|--------------------|----------------|--------------------|--------------------|
| REDDIT-B  | 2       | 1600           | 233                | 274                | 400            | 1212               | 1392               |
| REDDIT-5K | 5       | 4000           | 375                | 433                | 1000           | 1043               | 1246               |
| REDDIT-12K| 11      | 9507           | 258                | 277                | 2390           | 924                | 1066               |
| Citation  | 2       | 400            | 812                | 898                | 100            | 2886               | 4288               |
| Bitcoin   | 2       | 160            | 412                | 977                | 40             | 880                | 2996               |
| Q&A       | 4       | 800            | 918                | 1397               | 200            | 4295               | 5795               |
| Social    | 2       | 800            | 492                | 458                | 200            | 2713               | 2410               |


### **Table.2 Approximation error on datasets**

| Method   | Static Network      |                            |                     |                          |                     |                           |
|----------|---------------------|----------------------------|---------------------|--------------------------|---------------------|---------------------------|
|          | REDDIT-B            | REDDIT-B                   | REDDIT-5K           | REDDIT-5K                | REDDIT-12K          | REDDIT-12K                |
|          | $WD$                | $PIE$                      | $WD$                | $PIE$                    | $WD$                | $PIE$                     |
| GIN_PI   | -                   | 1.78e-03 $\pm$ 7.0e-04    | -                   | 2.20e-04 $\pm$ 4.3e-04    | -                   | 5.18e-04 $\pm$ 4.3e-04    |
| GAT_PI   | -                   | 1.57e-03 $\pm$ 3.5e-04    | -                   | 5.49e-04 $\pm$ 1.7e-04    | -                   | 7.82e-04 $\pm$ 2.2e-04    |
| GAT      | 0.910 $\pm$ 0.12    | 7.73e-03 $\pm$ 1.0e-02    | 0.731 $\pm$ 0.01    | 5.36e-04 $\pm$ 2.0e-04    | 0.794 $\pm$ 0.01    | 1.48e-03 $\pm$ 3.8e-04    |
| PDGNN    | 0.679 $\pm$ 0.29    | 2.91e-03 $\pm$ 3.0e-03    | 0.697 $\pm$ 0.04    | 5.10e-04 $\pm$ 2.2e-04    | 0.744 $\pm$ 0.03    | 5.10e-04 $\pm$ 2.2e-04    |
| TOGL     | 1.114 $\pm$ 0.19    | 1.82e-03 $\pm$ 5.0e-04    | 0.829 $\pm$ 0.08    | 1.95e-03 $\pm$ 7.3e-04    | 1.021 $\pm$ 0.04    | 1.95e-03 $\pm$ 7.3e-04    |
| RePHINE  | 0.816 $\pm$ 0.01    | 6.78e-04 $\pm$ 1.4e-05    | 0.523 $\pm$ 0.01    | 4.12e-04 $\pm$ 2.5e-04    | 0.685 $\pm$ 0.04    | 4.12e-04 $\pm$ 2.5e-04    |
| DNDN-EF  | 0.610 $\pm$ 0.05    | 7.68e-04 $\pm$ 1.3e-04    | 0.498 $\pm$ 0.07    | 3.78e-04 $\pm$ 4.2e-05    | 0.595 $\pm$ 0.03    | 3.78e-04 $\pm$ 4.2e-05    |
| DNDN     | **0.499 $\pm$ 0.01**| **1.56e-04 $\pm$ 1.5e-05**| **0.317 $\pm$ 0.05**| **5.21e-05 $\pm$ 1.5e-05**| **0.389 $\pm$ 0.05**| **6.73e-05 $\pm$ 1.6e-05**|

| Method   | Dynamic Network     |                           |                     |                           |                     |                           |                     |                            |
|----------|---------------------|---------------------------|---------------------|---------------------------|---------------------|---------------------------|---------------------|----------------------------|
|          | Citation            | Citation                  | Bitcoin             | Bitcoin                   | Q & A               | Q & A                     | Social              | Social                     |
|          | $WD$                | $PIE$                     | $WD$                | $PIE$                     | $WD$                | $PIE$                     | $WD$                | $PIE$                      |
| GIN_PI   | -                   | 4.71e-04 $\pm$ 1.6e-04    | -                   | 2.73e-03 $\pm$ 9.1e-04    | -                   | 5.15e-03 $\pm$ 1.8e-03    | -                   | 1.44e-03 $\pm$ 1.9e-04     |  
| GAT_PI   | -                   | 7.82e-04 $\pm$ 2.2e-04    | -                   | 1.93e-03 $\pm$ 3.6e-04    | -                   | 2.80e-03 $\pm$ 7.9e-04    | -                   | 9.04e-04 $\pm$ 1.1e-04     |
| GAT      | 0.960 $\pm$ 0.11    | 1.40e-03 $\pm$ 6.3e-03    | 2.508 $\pm$ 0.11    | 1.09e-01 $\pm$ 1.8e-01    | 3.185 $\pm$ 1.10    | 1.44e-01 $\pm$ 2.4e-01    | 0.900 $\pm$ 0.01    | 9.20e-04 $\pm$ 3.7e-04     |
| PDGNN    | 1.313 $\pm$ 0.44    | 1.87e-02 $\pm$ 2.0e-02    | 2.016 $\pm$ 0.44    | 6.81e-02 $\pm$ 9.4e-02    | 3.708 $\pm$ 1.74    | 4.13e-01 $\pm$ 4.5e-01    | 1.010 $\pm$ 0.16    | 2.34e-03 $\pm$ 3.4e-03     |
| TOGL     | 0.935 $\pm$ 0.07    | 2.45e-03 $\pm$ 2.0e-03    | 1.622 $\pm$ 0.07    | 2.12e-02 $\pm$ 3.3e-02    | 2.064 $\pm$ 0.19    | 8.73e-03 $\pm$ 3.2e-02    | 0.943 $\pm$ 0.04    | 1.54e-03 $\pm$ 1.3e-03     |
| RePHINE  | 0.775 $\pm$ 0.02    | 3.38e-04 $\pm$ 1.5e-04    | 1.867 $\pm$ 0.02    | 2.50e-02 $\pm$ 8.8e-03    | 2.270 $\pm$ 0.01    | 4.88e-02 $\pm$ 2.3e-02    | 0.703 $\pm$ 0.01    | 6.81e-04 $\pm$ 4.6e-04     |
| DNDN-EF  | 0.815 $\pm$ 0.01    | 3.98e-04 $\pm$ 4.7e-05    | 1.364 $\pm$ 0.01    | 8.28e-03 $\pm$ 1.0e-02    | 1.442 $\pm$ 0.23    | 1.22e-02 $\pm$ 1.1e-02    | 0.654 $\pm$ 0.12    | 3.15e-04 $\pm$ 3.1e-04     |
| DNDN     | **0.591 $\pm$ 0.02**| **1.29e-04 $\pm$ 3.1e-05**| **0.804 $\pm$ 0.02**| **1.33e-03 $\pm$ 7.4e-04**| **0.908 $\pm$ 0.04**| **2.13e-03 $\pm$ 2.1e-04**| **0.514 $\pm$ 0.03**| **1.01e-04 $\pm$ 1.37e-05**|

### **Table.3 Transferability across graph datasets of varying sizes (WD)**

| Method     | Static Network   |                  |                  |Dynamic Network   |                  |                  |                  |
|------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|
|            | REDDIT-B         | REDDIT-5K        | REDDIT-12K       | Citation         | Bitcoin          | Q & A            | Social           | 
| Standard   | 0.553 $\pm$ 0.01 | 0.488 $\pm$ 0.04 | 0.409 $\pm$ 0.12 | 0.838 $\pm$ 0.05 | 0.849 $\pm$ 0.04 | 1.355 $\pm$ 0.12 | 0.737 $\pm$ 0.01 |
| Pre\_train | 0.438 $\pm$ 0.01 | 0.177 $\pm$ 0.01 | 0.183 $\pm$ 0.01 | 0.850 $\pm$ 0.05 | 0.924 $\pm$ 0.02 | 1.268 $\pm$ 0.02 | 0.739 $\pm$ 0.01 |
| Fine\_tune | 0.424 $\pm$ 0.01 | 0.173 $\pm$ 0.01 | 0.176 $\pm$ 0.01 | 0.704 $\pm$ 0.01 | 0.831 $\pm$ 0.01 | 1.121 $\pm$ 0.02 | 0.708 $\pm$ 0.01 |

### **Table.4 Training Time Comparison(s/epoch)**

| Method  | Static Network |           |            |Dynamic Network |         |       |       |
|---------|----------------|-----------|------------|----------------|---------|-------|-------|
|         | REDDIT-B       | REDDIT-5K | REDDIT-12K | Citation       | Bitcoin | Q & A |Social | 
| GIN_PI  | 27             | 88        | 173        | 6.9            | 6.2     | 37    | 8.7   |
| GAT_PI  | 40             | 108       | 257        | 11             | 8.8     | 41    | 19    |
| GAT     | 363            | 1795      | 2714       | 168            | 165     | 1426  | 134   |
| PDGNN   | 364            | 1692      | 2667       | 165            | 77      | 1415  | 129   |
| TOGL    | 190            | 797       | 1320       | 77             | 71      | 583   | 74    |
| RePHINE | 471            | 1456      | 2691       | 188            | 146     | 1054  | 165   |
| DNDN    | 197            | 856       | 1446       | 55             | 69      | 456   | 58    |


### **Table.5 Average Time Comparison for Generating Persistence Diagrams**

| Method | Static Network    |                  |                  | Dynamic Network  |                       |                  |                  |
|--------|-------------------|------------------|------------------|------------------|-----------------------|------------------|------------------|
|        | REDDIT-B          | REDDIT-5K        | REDDIT-12K       | Citation         | Q & A                | Bitcoin          | Social           |
| GUDHI  | 16.46 $\pm$ 0.001 | 4.64 $\pm$ 0.002 | 5.10 $\pm$ 0.002 | 1.37 $\pm$ 0.001 | 1.33 $\pm$ 0.002      | 3.73 $\pm$ 0.001 | 4.85 $\pm$ 0.002 |
| DNDN   | 1.69 $\pm$ 0.003  | 0.66 $\pm$ 0.003 | 0.70 $\pm$ 0.003 | 0.13 $\pm$ 0.003 | 0.14 $\pm$ 0.003      | 0.17 $\pm$ 0.001 | 0.31 $\pm$ 0.003 |


### **Table.6 Total Time for Line Graph Transformation**

|                            | Static Network |           |            |Dynamic Network |         |       |        |
|----------------------------|----------------|-----------|------------|----------------|---------|-------|--------|
|                            | REDDIT-B       | REDDIT-5K | REDDIT-12K | Citation       | Bitcoin | Q & A | Social | 
| Line graph transformation  | 318            | 486       | 965        | 58             | 32      | 149   | 58     |


### **Table.7 Dynamic Graph Classification Results**

Due to time constraints, some experiments are still running. Results will be updated subsequently.

| Method         | Static Network |             |             |             | Dynamic Network |             |             |             |             |             |             |             |
|----------------|----------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
|                | REDDIT-B       | REDDIT-B    | REDDIT-5K   | REDDIT-5K   | Citation    | Citation    | Q$\&$A      | Q$\&$A      | Bitcoin     | Bitcoin     | Social      | Social      |
|                | Small          | Large       | Small       | Large       | Small       | Large       | Small       | Large       | Small       | Large       | Small       | Large       |
| GCN            | 73.8 $\pm$ 0.5 | 50.1 $\pm$ 0.1| 33.0 $\pm$ 0.3| 20.7 $\pm$ 0.0| 50.0 $\pm$ 0.0| 52.5 $\pm$ 0.7| 65.0 $\pm$ 0.8| 49.5 $\pm$ 0.0| 84.4 $\pm$ 1.1| 52.3 $\pm$ 0.3| 84.0 $\pm$ 0.6| 88.2 $\pm$ 0.2|
| GAT            | 79.4 $\pm$ 1.2   | 52.1 $\pm$ 0.0| 38.1 $\pm$ 0.2| 20.0 $\pm$ 0.0| 51.2 $\pm$ 0.1| 49.0 $\pm$ 0.1| 41.9 $\pm$ 0.2| 57.0 $\pm$ 0.4| 87.5 $\pm$ 1.2| 50.1 $\pm$ 0.3| 71.3 $\pm$ 1.2| 81.3 $\pm$ 0.6|
| GraphSage      | 79.1 $\pm$ 0.4   | 52.3 $\pm$ 0.2| 33.2 $\pm$ 0.2| 21.5 $\pm$ 0.0| 67.5 $\pm$ 0.3| 56.0 $\pm$ 0.1| 76.3 $\pm$ 0.7| 53.0 $\pm$ 0.4| 78.1 $\pm$ 0.1| 62.3 $\pm$ 0.3| 91.8 $\pm$ 1.5| 89.5 $\pm$ 1.2|
| GIN            | 73.8 $\pm$ 0.2   | 51.1 $\pm$ 0.0| 20.9 $\pm$ 0.0| 25.5 $\pm$ 0.1| 83.8 $\pm$ 0.3| 51.1 $\pm$ 0.6| 70.0 $\pm$ 1.2| 68.0 $\pm$ 1.8| 68.8 $\pm$ 1.3| 71.2 $\pm$ 1.2| 81.3 $\pm$ 0.6| 78.5 $\pm$ 0.6|
| DySAT          | 78.3 $\pm$ 0.7   | 54.5 $\pm$ 0.0| 32.2 $\pm$ 0.1| 22.3 $\pm$ 0.0|86.9 $\pm$ 1.3| 63.0 $\pm$ 1.6| 78.4 $\pm$ 2.5| 77.3 $\pm$ 2.7| 87.4 $\pm$ 1.4| 72.2 $\pm$ 0.7| 91.2 $\pm$ 1.5| 75.2 $\pm$ 2.0|
| DHGAT          | 76.4 $\pm$ 0.1   | 56.2 $\pm$ 0.1| 36.7 $\pm$ 0.2| 21.7 $\pm$ 0.0| 85.8 $\pm$ 1.2| 65.9 $\pm$ 0.3| 76.7 $\pm$ 0.1| 78.4 $\pm$ 0.2| 86.4 $\pm$ 0.3| 65.9 $\pm$ 0.3| 92.3 $\pm$ 0.4| 86.5 $\pm$ 0.2|
| EvolveGCN-O    | 82.2 $\pm$ 0.4   | 61.0 $\pm$ 0.0| 34.8 $\pm$ 0.1| 30.1 $\pm$ 0.1| 84.5 $\pm$ 0.8| 63.4 $\pm$ 0.3| 79.2 $\pm$ 0.4| 78.3 $\pm$ 0.6| 89.3 $\pm$ 0.6| 73.3 $\pm$ 0.2| 89.2 $\pm$ 0.4| 81.2 $\pm$ 0.9|
| EvolveGCN-H    | 82.5 $\pm$ 0.5   | 59.5 $\pm$ 0.2| 32.5 $\pm$ 0.0| 28.3 $\pm$ 0.0| 86.7 $\pm$ 0.2| 62.4 $\pm$ 0.5| 80.4 $\pm$ 0.5| 80.2 $\pm$ 0.6| **89.6 $\pm$ 0.4**| 75.5 $\pm$ 0.6| 88.6 $\pm$ 1.2| 80.5 $\pm$ 1.4|
| Roland         | **84.6 $\pm$ 0.2**| 65.2 $\pm$ 0.2|   |  |**87.5 $\pm$ 0.2**| 68.2 $\pm$ 0.7|78.5 $\pm$ 0.6| 78.6 $\pm$ 0.4| 80.2 $\pm$ 0.2| 72.1 $\pm$ 0.5| 90.5 $\pm$ 0.2|**91.2 $\pm$ 0.3**|
| DNDN           | 83.4 $\pm$ 0.2| **73.6 $\pm$ 1.2**| **56.7 $\pm$ 0.4**| **41.3 $\pm$ 1.0**| 85.6 $\pm$ 0.2| **72.4 $\pm$ 0.4**| **83.4 $\pm$ 0.4**| **81.2 $\pm$ 0.5**| 84.5 $\pm$ 0.5| **77.6 $\pm$ 1.0**| **94.5 $\pm$ 0.2**| 89.2 $\pm$ 0.1 |

## Hyperparameter Tuning

For detailed settings, please refer to the `config.yml` file. Specifically for DNDN, we set the range for network layers to [2, 3, 4, 5, 6], learning rates to [0.01, 0.001, 0.0001, 0.00001], and hidden layer dimensions to [32, 64, 128]. We used grid search for hyperparameter tuning, ultimately selecting 5 layers, a learning rate of 0.001, and a hidden dimension of 32. We did not perform fine-tuned parameter adjustments for each dataset. For other methods, we kept the number of layers and hidden dimensions fixed and adjusted the learning rate. Additionally, for dynamic GNNs, we processed the dynamic graph into k snapshots, with k ranging from [8, 10, 15, 20], and finally chose 10 as the number of snapshots.

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

Please unzip the data before use.

```
7z x ./DNDN/data/SocioPatternsDataset/GCB_citation_dowker.7z
```

```
python main.py
```
You can modify the necessary configurations in 'config.yml'.

## Custom Datasets

'utils.process_data' is the method we provide for processing your custom datasets. Its main functionalities include generating dynamic Dowker filtration ground truths and constructing dynamic graph classification datasets, etc. You can refer to the function 'construct_sp_dataset()' for constructing custom datasets.
