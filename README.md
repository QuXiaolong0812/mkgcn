# MKGCN4M

This repository is the implementation of a multimodal Recommendation system:
> MKGCN4M: Multi-modal Knowledge Graph Convolutional Networks for Music Recommendation System 

Designed by Qu Xiaolong in Beijing Forestry University.


### Files in the folder

- `data/`
  - `movie-d/`
    - `itemID2entityID.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
    - `userID2itemID4ratings.csv`: raw rating file of MovieLens-1M;
    - `modals/`: 7 modals files for music-d dataset;
    - `data_config.py`: dataset configuration file;
- `other .py files`: implementations of MKGCN4M.
>Attention: music-d is multi-modal music dataset by myself. Actually, we have created four multimodal 
> music datasets of different sizes, but due to the need for data copyright and confidentiality,
> we only open the smallest one.


### Required packages
The code has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):
- pytorch == 1.1.0
- numpy == 1.14.5
- sklearn == 0.19.1


### Running the code
```
$ python main.py --dataset music-d (note: use -h to check optional arguments)
```
### Note
>We will update more information in the future, please stay tuned.
