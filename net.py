import os

import numpy as np

import settings
from utils import add_caffe_to_path


class CaffeNetClassifier():
    """This class is a wrapper around a CNN defined in Caffe which can predict
    class labels with softmax."""
    def __init__(
        self,
        deployfile_relpath,
        weights_relpath,
        image_dims=(256, 256),
        mean=(104, 117, 123),
        device_id=0,
        input_scale=1,
    ):
        self._net, self._caffe = _load_caffe_net(
            deployfile_relpath=deployfile_relpath,
            weights_relpath=weights_relpath,
            image_dims=image_dims,
            mean=mean,
            device_id=device_id,
            input_scale=input_scale,
        )

    def predict(self, filename, oversample=True):
        if isinstance(filename, np.ndarray):
            inputs = [filename]
        else:
            inputs = [self._caffe.io.load_image(filename)]
        return self._net.predict(inputs, oversample=oversample, auto_reshape=True)

    def extract_features(self, filename, blob_names, oversample=True):
        # sanity checking
        if len(set(blob_names)) != len(blob_names):
            raise ValueError("Duplicate name in blob_names: %s" % blob_names)

        self.predict(filename, oversample=oversample)
        ret = {}
        for blob_name in blob_names:
            blob_data = self._net.blobs[blob_name].data.copy()
            if oversample:
                orig_shape = blob_data.shape
                blob_data = blob_data.reshape((len(blob_data) / 10, 10, -1))
                blob_data = blob_data.mean(1)
                blob_data = blob_data.reshape((orig_shape[0]/10,) + orig_shape[1:])
            ret[blob_name] = blob_data

        return ret


def _load_caffe_net(
    deployfile_relpath,
    weights_relpath,
    image_dims=(256, 256),
    mean=(104, 117, 123),
    device_id=0,
    input_scale=1,
):
    add_caffe_to_path()
    import caffe

    mean = np.array(mean)

    model_file = os.path.join(settings.CAFFE_ROOT, deployfile_relpath)
    pretrained_file = os.path.join(settings.CAFFE_ROOT, weights_relpath)

    if settings.CAFFE_GPU:
        print 'Using GPU'
        caffe.set_mode_gpu()
        print 'Using device #{}'.format(device_id)
        caffe.set_device(device_id)
    else:
        print 'Using CPU'
        caffe.set_mode_cpu()

    net = caffe.Classifier(
        model_file=model_file,
        pretrained_file=pretrained_file,
        mean=mean,
        channel_swap=(2, 1, 0),
        raw_scale=255,
        input_scale=input_scale,
        image_dims=image_dims,
    )

    return net, caffe
