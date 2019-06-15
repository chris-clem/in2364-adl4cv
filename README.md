# One Shot Video Object Segmentation with GCN-based contour propagation
> If you have any question or want to report a bug, please open an issue.

## Repository Structure
- **download_DAVIS_2016.sh**: Downloads DAVIS 2016 dataset and saves it to **DAVIS_2016**
- **create_contours.ipynb**: Saves contour points of DAVIS 2016 annotations as numpy arrays
- **create_translations.ipynb**: Saves translations between points of two consecutive contours as numpy arrarys
- **gcn_davis_2016.ipynb**: Creates a custom PyTorch Geometric InMemoryDataset for the DAVIS 2016 dataset and trains a simple GCN.
- **DAVIS_2016**: Folder in which DAVIS 2016 dataset is stored
- **Experiments**: Folder in which PyTorch Geometric tutorials are stored
- **Images**: Folder in which images used in notebooks are stored
- **OSVOS-PyTorch**: Original OSVOS-PyTorch implementation
- **PyTorch_Geometric_Datasets**: Folder to store PG DAVIS 2016 dataset

## Dependencies
- See spec-file_linux.txt for condaenv for ubuntu 16.04
- See spec-file_win64.txt for condaenv for windows 10


## ToDos
- [x] Check which sequences to use [Christoph]
```python
SKIP_SEQUENCES = ['bmx-trees', 'bus', 'cows', 'dog-agility', 'horsejump-high', 
                  'horsejump-low', 'kite-walk', 'lucia', 'libby', 'motorbike',
                  'paragliding', 'rhino', 'scooter-gray', 'swing']
```
- [ ] Extract OSVOS feature vectors [Max]
- [x] Create PyTorch Geometric Dataset
- [x] Decide which GCN implementation to use: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/dgcnn_classification.py 
- [x] Define train and val sequences [Christoph]
```python
TRAIN_SEQUENCES = ['bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 
                   'car-turn', 'dance-jump', 'dog-agility', 'drift-turn', 
                   'elephant', 'flamingo', 'hike', 'hockey', 'horsejump-low', 
                   'kite-walk', 'lucia', 'mallard-fly', 'mallard-water', 
                   'motocross-bumps', 'motorbike', 'paragliding', 'rhino', 
                   'rollerblade', 'scooter-gray', 'soccerball', 'stroller',
                   'surf', 'swing', 'tennis', 'train']

VAL_SEQUENCES = ['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout',
                 'car-shadow', 'cows', 'dance-twirl', 'dog', 'drift-chicane', 
                 'drift-straight', 'goat', 'horsejump-high', 'kite-surf', 'libby', 
                 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black', 
                 'soapbox']
```
