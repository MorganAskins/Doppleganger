# Doppleganger
One-to-many facial matching program using feature triplets without labeled attributes (learned
feature vectors only). For a given input image, the network computes a set of features and then
searches a celebrity database for the images that most-similarly match the input image. The dataset
used in the example is not included in the reposity, but a new set can be generated with the provided
scripts (and data caching is performed to ease computation after the first initial batching).

## Install
`requirements.txt` contains all dependencies. A manual fix to vggface is required:  
See: https://stackoverflow.com/questions/68862735/keras-vggface-no-module-named-keras-engine-topology
In `lib/python3.x/site-packages/keras_vggface/model.py` change  
```
from keras.engine.topology import get_source_inputs
```
to  
```
from keras.utils.layer_utils import get_source_inputs
```
