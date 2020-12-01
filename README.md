# mano_pybullet
[MANO](http://mano.is.tue.mpg.de/)-based hand models for the [PyBullet](https://pybullet.org/wordpress/) simulator.

## Install

### From source code

```
git clone https://github.com/ikalevatykh/mano_pybullet.git
cd mano_pybullet
python setup.py develop 
```
or
```
python setup.py install
```

### Run tests

```
export MANO_MODELS_DIR='/path/to/mano_v*_*/models/'
pytest
```

## Download MANO models

- Register at the [MANO website](http://mano.is.tue.mpg.de/) and download the models.
- Unzip the file mano_v*_*.zip: `unzip mano_v*_*.zip`
- Set environment variable: `export MANO_MODELS_DIR=/path/to/mano_v*_*/models/`


## Hand models

The package provides MANO-based rigid hand model.

![TeaTime](https://github.com/ikalevatykh/mano_pybullet/blob/master/media/tea_time.gif?raw=true "TeaTime")

## Gym environments

The package also provides several [OpenAI gym](https://gym.openai.com/) environments:

- HandEnv - base environment with one or two hands on the scene.

- HandObjectEnv - base environment with hands and an object on the scene.

- HandLiftEnv - environment where the target is to lift an object.

![HandLiftEnv](https://github.com/ikalevatykh/mano_pybullet/blob/master/media/lift_duck.gif?raw=true "HandLiftEnv")

- HandPushEnv - environment where the target is to push an object.

![HandPushEnv](https://github.com/ikalevatykh/mano_pybullet/blob/master/media/push_teddy.gif?raw=true "HandPushEnv")


## Graphical debugging

```
python -m mano_pybullet.tools.gui_control
```

usage: GUI debug tool [-h] [--dofs DOFS] [--left-hand] [--right-hand] [--visual-shapes] [--no-visual-shapes] [--self-collisions] [--no-self-collisions]

optional arguments:
-  -h, --help            show help message and exit
-  --dofs DOFS           number of degrees of freedom (20 or 45) [default=20]
-  --left-hand           show left hand
-  --right-hand          show right hand [default]
-  --visual-shapes       show visual shapes [default]
-  --no-visual-shapes    hide visual shapes
-  --self-collisions     enable self collisions [default]
-  --no-self-collisions  disable self collisions

## Citation
If you find mano_pybullet useful in your research, please cite the repository using the following BibTeX entry.
```
@Misc{kalevatykh2020mano_pybullet,
  author =       {Kalevatykh, Igor et al.},
  title =        {mano_pybullet - porting the MANO hand model to the PyBullet simulator},
  howpublished = {Github},
  year =         {2020},
  url =          {https://github.com/ikalevatykh/mano_pybullet}
}
```
## License
mano_pybullet is released under the [GPLv3](https://github.com/ikalevatykh/mano_pybullet/blob/master/LICENSE).