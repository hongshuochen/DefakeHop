# DefakeHop: A Light-Weight High-Performance Deepfake Detector

![Framework](img/framework.png)

## Required packages
```bash
conda install -c conda-forge opencv
conda install -c conda-forge xgboost
conda install -c anaconda scikit-image
conda install -c conda-forge matplotlib
```

## Preprocessing
- Extracting the facial landmarks using [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)
- Face alignment
- Crop facial regions

## How to run
```bash
python model.py
```

## Cite us
If you use this repository, please consider to cite.
```
@misc{chen2021defakehop,
      title={DefakeHop: A Light-Weight High-Performance Deepfake Detector}, 
      author={Hong-Shuo Chen and Mozhdeh Rouhsedaghat and Hamza Ghani and Shuowen Hu and Suya You and C. -C. Jay Kuo},
      year={2021},
      eprint={2103.06929},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
