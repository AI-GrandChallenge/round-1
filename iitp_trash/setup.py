#nsml: nsml/default_ml:cuda9_torch1.0
from distutils.core import setup

setup(
    name='iitp_trash',
    version='1.0',
    install_requires=[
                      'tqdm',
                      'pillow',
                      ]
)
