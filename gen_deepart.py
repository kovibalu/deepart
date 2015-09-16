import numpy as np

from fet_extractor import load_fet_extractor
from deepart import gen_target_data, optimize_img
from test_deepart import test_all_gradients


def setup_classifier():
    #deployfile_relpath = 'models/VGG_CNN_19/VGG_ILSVRC_19_layers_deploy_deepart.prototxt'
    #weights_relpath = 'models/VGG_CNN_19/VGG_ILSVRC_19_layers.caffemodel'
    #image_dims = (1014/2, 1280/2)
    #mean = (104, 117, 123)

    deployfile_relpath = 'models/VGG_CNN_19/VGG_ILSVRC_19_layers_deploy_fullconv.prototxt'
    weights_relpath = 'models/VGG_CNN_19/vgg_normalised.caffemodel'
    image_dims = (333, 500)
    mean = (103.939, 116.779, 123.68)
    device_id = 0
    input_scale = 1.0

    caffe, net = load_fet_extractor(
        deployfile_relpath, weights_relpath, image_dims, mean, device_id,
        input_scale
    )

    return caffe, net, image_dims


def deepart():
    np.random.seed(123)

    root_dir = 'results'
    display = 100
    max_iter = 2000
    # list of targets defined by tuples of
    # (
    #     image path,
    #     target blob names (these activations will be included in the loss function),
    #     whether we use style (gram) or content loss,
    #     weighting factor
    # )
    targets = [
        ('images/starry_night.jpg', ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'], True, 100),
        ('images/tuebingen.jpg', ['conv4_2'], False, 1),
    ]
    # These have to be in the same order as in the network!
    all_target_blob_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']

    caffe, net, image_dims = setup_classifier()

    # Generate activations for input images
    target_data_list = gen_target_data(root_dir, caffe, net, targets)

    # Generate white noise image
    init_img = np.random.normal(loc=0.5, scale=0.1, size=image_dims + (3,))

    solver_type = 'L-BFGS-B'
    solver_param = {}

    #test_all_gradients(init_img, net, all_target_blob_names, targets, target_data_list)

    optimize_img(
        init_img, solver_type, solver_param, max_iter, display, root_dir, net,
        all_target_blob_names, targets, target_data_list
    )


if __name__ == '__main__':
    deepart()
