# THIS REPOSITORY IS A CLONE FROM konstantg W/ MINOR CHANGES TO DIRECTLY RUN THE CODE AND CLONE AN ERROR
# ALL RIGHTS RESERVED TO THE CITED PERSONS BELOW


## Void Filling of Digital Elevation Models with Deep Generative Models

This GitHub repository implements and evaluates the method described in the paper [1], which is an adaptation to the context of Digital Elevation Models (DEMs) from the method DeepFill described in [2]. In addition pre-trained models are provided, as well as the DEMs used for the evaluation of the method.

[1] K. Gavriil, O.J.D. Barrowclough, G. Muntingh, _Void Filling of Digital Elevation Models with Deep Generative Models_, in IEEE Geoscience and Remote Sensing Letters, vol. 16, no. 10, pp. 1645-1649, Oct. 2019. [[DOI](https://doi.org/10.1109/LGRS.2019.2902222)] [[ArXiv](https://arxiv.org/abs/1811.12693)]

[2] J. Yu, Z. Lin, J. Yang, X. Shen, X. Lu, and T. S. Huang, _Generative image inpainting with contextual attention_, in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

---

<p>
<img src="https://user-images.githubusercontent.com/26650959/49333780-24d37380-f5c5-11e8-8dc9-104ac7373874.png" width="24.5%" />
<img src="https://user-images.githubusercontent.com/26650959/49333779-24d37380-f5c5-11e8-88de-b1157b59537c.png" width="24.5%" />
<img src="https://user-images.githubusercontent.com/26650959/49333784-256c0a00-f5c5-11e8-8868-953bc154b182.png" width="24.5%" />
<img src="https://user-images.githubusercontent.com/26650959/49333783-256c0a00-f5c5-11e8-9243-a454924bb0af.png" width="24.5%" />
</p>

<p>
<img src="https://user-images.githubusercontent.com/26650959/49333787-269d3700-f5c5-11e8-8586-fcd9bfdc0768.png" width="24.5%" />
<img src="https://user-images.githubusercontent.com/26650959/49333788-269d3700-f5c5-11e8-8d6c-6702063abcdf.png" width="24.5%" />
<img src="https://user-images.githubusercontent.com/26650959/49333785-2604a080-f5c5-11e8-8043-38df141459dc.png" width="24.5%" />
<img src="https://user-images.githubusercontent.com/26650959/49333786-2604a080-f5c5-11e8-9fab-2d5f74cca6c8.png" width="24.5%" />
</p>

Selection of results of the DeepDEMFill void filling method for Digital Elevation Models.

---

## Setup

* Install python3.
* Install [tensorflow](https://www.tensorflow.org/install/) (tested on Release 1.3.0, 1.4.0, 1.5.0, 1.6.0, 1.7.0).
* Install tensorflow toolkit [neuralgym](https://github.com/konstantg/neuralgym) (run `pip install git+https://github.com/konstantg/neuralgym`).
* Clone the repository `git clone https://github.com/konstantg/dem-fill.git`

Other dependencies: [`opencv-python`](https://pypi.org/project/opencv-python/), [`PyYAML`](https://pypi.org/project/PyYAML/)

## Testing pretrained models

[Norway Landscape](https://drive.google.com/open?id=1OaQ0PNqaLQ5-TVerRzQl3OkIXdf9R4US) | [Norway Cities](https://drive.google.com/open?id=1Y_noNEPh6Uzm1OFzGWT3Tsey83nYsgCl)

Download the desired model(s), create a `model_logs/` directory in the `dem-fill/` directory and extract the zip folder there. Specifically, to run the examples below, the extracted files (`checkpoint` etc.) should be in either the `model_logs/norway_land/` or the `model_logs/norway_cities/` directory respectively.

Model `norway_land` was trained on 10m-resolution DEMs of Western and Eastern Norway while `norway_cities` was trained on 2m-resolution DEMs of the three largest cities in Norway, namely Oslo, Trondheim, and Bergen. The input in both cases are DEMs of size 256x256. The size of the void ranges from 64x64 up to 128x128 (not necessarily rectangular) and is randomly placed over the DEM.

To run:

```bash
# Norway Landscape
python test.py --image data/land01.tif --mask data/land01mask.png --output data/land01out.tif --checkpoint_dir model_logs/norway_land/

# Norway Cities
python test.py --image data/city01.tif --mask data/city01mask.png --output data/city01out.tif --checkpoint_dir model_logs/norway_cities/
```

__Note:__ If you do not have a TIF viewer, [IrfanView](https://www.irfanview.com/) is recommended for Windows.

## Results

We compared our results with two other void-filling methods, namely inverse distance weighting (solution provided by `gdal_fillnodata.py`) and LR B-spline interpolation. The 50 random urban and 50 random rural DEMs, along with the randomly generated masks and the void-filling results, used in the comparison can be downloaded [here](https://drive.google.com/open?id=12GaxRnzoJUJNKdXn-yIeWl1mwpjaz8DO).

## Training

For training on your DEMs, please modify accordingly the `data_from_fnames.py` file from the installed `neuralgym` package to accept your files. For example, it was more efficient to use an [`h5py`](https://www.h5py.org/) data structure to store and sample the Norway landscape dataset, while for the Norway cities dataset we preload the Oslo, Bergen and Trondheim DEMs to memory for sampling. Please refer to the `data_from_fnames.py` file for these examples.

1. Training:
    * Modify the data `data_from_fnames.py` file accordingly.
    * Modify [inpaint.yml](/inpaint.yml) to set LOG_DIR, IMG_SHAPES and other parameters.
    * Run `python train.py`.
2. Resume training:
    * Modify MODEL_RESTORE flag in [inpaint.yml](/inpaint.yml). E.g., MODEL_RESTORE: 20181119115621405562_norway_cities.
    * Run `python train.py`.
3. Testing:
    * Run `python test.py --image examples/input.tif --mask examples/mask.png --output examples/output.tif --checkpoint model_logs/your_model_dir`.


## License

CC 4.0 Attribution-NonCommercial International

The software is for educational and academic research purpose only.

## Acknowledgements

We adapted the GitHub repository [generative_inpainting](https://github.com/JiahuiYu/generative_inpainting) to the setting of Digital Elevation Models. The open source C++ library [GoTools](https://github.com/SINTEF-Geometry/GoTools) was used for generating the LR B-spline data. Data provided courtesy Norwegian Mapping Authorities (www.hoydedata.no), copyright Kartverket (CC BY 4.0). This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 675789. This projected was also supported by an IKTPLUSS grant, project number 270922, from the Research Council of Norway.

## Citing

```
@article{8669876,
author={K. {Gavriil} and G. {Muntingh} and O. J. D. {Barrowclough}},
journal={IEEE Geoscience and Remote Sensing Letters},
title={Void Filling of Digital Elevation Models With Deep Generative Models},
year={2019},
volume={16},
number={10},
pages={1645-1649},
doi={10.1109/LGRS.2019.2902222},
ISSN={1558-0571},
month={Oct},
}
```
