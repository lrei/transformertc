# TransformerTC
[![Documentation Status](https://readthedocs.org/projects/transformertc/badge/?version=latest)](http://transformertc.readthedocs.io/?badge=latest)

## Introductions
Transformer based Token Classification.

Currently supports training and inference for Named Entity Recognition (NER)
tasks using BERT models.

## Examples

See ```examples/``` directory.


## Install
This is a python package with `setupy.py`.

Tip: It’s probably a good idea to install numpy, scikit-learn, and pytorch
using  your preferred installation packages (i.e. optimized for your CPU/GPU).


Example using virtualenv:

```bash
virtualenv venv
source venv/bin/activate
cd transformertc
pip install -r requirements.exact.txt
python setup.py install
```

Note: Two different `requirements.txt` files are provided:

   1. `requirements.txt` - just the required python packages;
   2. `requirements.exact.txt` - the pinned versions of the required packages.

If installing everything inside the virtualenv, there should be no problem
getting the exact versions of the packages. However, if for some reason the
environment is shared,  using the pinned versions might be restrictive.


### Running tips:
To disable debugging / asserts set the environment variable `PYTHONOPTIMIZE`
e.g.:

```bash
export PYTHONOPTIMIZE=TRUE
```

or  use `-O` e.g.

```bash
python -Oc program.py
```
