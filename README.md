# DefakeHop: A Light-Weight High-Performance Deepfake Detector

This is the official Python implementation of our work: "DefakeHop: A Light-Weight High-Performance Deepfake Detector" accepted at ICME 2021.

State-of-the-art Deepfake detection methods are built upon deep neural networks. In this work, we proposed a non deep learning method to detect Deepfake videos which use the successive subspace learning (SSL) principle to extract features from various parts of face images. We also use feature distillation module to further extract concise representation of the fake and real faces.

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
