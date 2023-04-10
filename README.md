# MKGCN4M

This repository is the implementation of the MKGCN:
> MKGCN: Multi-modal Knowledge Graph Convolutional Networks for Music Recommendation System 

![](https://static.qinux.top/mkgcn4m_github/framework-mkgcn4m.png)
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
>Attention: music-d is multi-modal music dataset by myself. Actually, I have created four multimodal 
> music datasets of different sizes, but due to the need for data copyright and confidentiality,
> I only open the smallest one.

Multimodal files definition are shown as below:

![](https://static.qinux.top/mkgcn4m_github/modals_explanation.png)

### Required packages
The code has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):
- pytorch == 2.0.0
- numpy == 1.14.5
- sklearn == 0.24.2


### Running the code
```
$ python main.py --dataset music-d (note: use -h to check optional arguments)
```
### Note
>I will update more information in the future, please stay tuned.
