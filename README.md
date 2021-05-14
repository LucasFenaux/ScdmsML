# ScmdsML
Code base for single scatter vs multiple scatter discrimination work for SCDMS

- The lstm.ipynb python notebook is a tutorial presenting the work in this repository on the raw data analysis performed. You should check it out first to learn about the work that's been done and what still needs to be worked on/improved.

## Structure
### Run scripts
This repository contains 2 different types of work:
1. Work on photoneutron raw data (most recent work), the following run scripts use raw data:
   - run_raw_all_channels.py
   - run_raw_1_att.py
   - run_benchmark.py
2. Work on RQ/RRQ (processed) data, the following run scripts use RQ/RRQ data: 
   - run_bdt.py
   - run_correlation.py
   - run_k_clustering.py
   - run_kmeans_scripts.py
   - run_NN.py
   - run_optics_sscripts.py
   - run_other_scripts.py
   - run_pca.py
   - run_wimp_classification.py
### Archives
The most recent run_raw_all_channels run have been archived for multiple different models and the training results/outputs can be found in the archive folder

## Dependencies 
- numpy: 1.19.2
- Pillow: 7.2.0
- torch: 1.7.1
- torchvision: 0.8.2
- uproot: 3.12.0
- sklearn: 0.0
- matplotlib: 3.3.2
- pandas: 1.2.0
- requests: 2.25.1
- tensorboard: 2.4.1
