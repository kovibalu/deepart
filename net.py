import os

import numpy as np

import settings
from utils import add_caffe_to_path


def load_caffe_net(
    deployfile_relpath,
    weights_relpath,
    image_dims=(256, 256),
    mean=(104, 117, 123),
    device_id=0,
    input_scale=1,
):
    add_caffe_to_path()
    import caffe

    class DeepArtNet(caffe.Net):
        """
        DeepArtNet extends Net for artistic pleasure.

        Parameters
        ----------
        mean, input_scale, raw_scale, channel_swap: params for
            preprocessing options.
        """
        def __init__(self, model_file, pretrained_file, image_dims,
                    mean=None, input_scale=None, raw_scale=None,
                    channel_swap=None):
            caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

            # configure pre-processing
            in_ = self.inputs[0]
            self.transformer = caffe.io.Transformer(
                {in_: self.blobs[in_].data.shape})
            self.transformer.set_transpose(in_, (2, 0, 1))
            if mean is not None:
                self.transformer.set_mean(in_, mean)
            if input_scale is not None:
                self.transformer.set_input_scale(in_, input_scale)
            if raw_scale is not None:
                self.transformer.set_raw_scale(in_, raw_scale)
            if channel_swap is not None:
                self.transformer.set_channel_swap(in_, channel_swap)

            self.image_dims = image_dims

        def preprocess_inputs(self, inputs, auto_reshape=True):
            """
            Preprocesses inputs.

            Parameters
            ----------
            inputs : iterable of (H x W x K) input ndarrays.

            Returns
            -------
            caffe_in: Preprocessed input which can be passed to forward.
            """
            # Scale to standardize input dimensions.
            input_ = np.zeros(
                (len(inputs),
                self.image_dims[0],
                self.image_dims[1],
                inputs[0].shape[2]),
                dtype=np.float32
            )
            for ix, in_ in enumerate(inputs):
                input_[ix] = caffe.io.resize_image(in_, self.image_dims)

            # Run net
            caffe_in = np.zeros(
                np.array(input_.shape)[[0, 3, 1, 2]],
                dtype=np.float32
            )
            if auto_reshape:
                self.reshape_by_input(caffe_in)

            for ix, in_ in enumerate(input_):
                caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)

            return caffe_in

        def reshape_by_input(self, caffe_in):
            """
            Reshapes the whole net according to the input
            """
            in_ = self.inputs[0]
            self.blobs[in_].reshape(*caffe_in.shape)
            self.transformer.inputs = {in_: self.blobs[in_].data.shape}
            self.reshape()

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

    net = DeepArtNet(
        model_file=model_file,
        pretrained_file=pretrained_file,
        mean=mean,
        channel_swap=(2, 1, 0),
        raw_scale=255,
        input_scale=input_scale,
        image_dims=image_dims,
    )

    return caffe, net

