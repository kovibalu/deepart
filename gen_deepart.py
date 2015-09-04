import os

import numpy as np
import skimage

import settings
from net import load_caffe_net
from utils import Timer, ensuredir, plot_and_save_2D_array


def get_imagenet_labels():
    imagenet_labels_filename = os.path.join(settings.CAFFE_ROOT, 'data/ilsvrc12/synset_words.txt')
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

    return labels


def save_loss_proc(filepath, loss_proc, last_x=-1):
    loss_proc_arr = [[i, loss] for i, loss in enumerate(loss_proc)]
    loss_proc_arr = np.array(loss_proc_arr)

    if last_x != -1:
        loss_proc_arr = loss_proc_arr[::-1][:last_x][::-1]

    plot_and_save_2D_array(
        filepath,
        loss_proc_arr,
        xlabel='Iteration',
        ylabel='Content Loss',
    )


def save_image_blob(filepath, net, data):
    deproc_img = net.transformer.deprocess(net.inputs[0], data)
    deproc_img = np.clip(deproc_img, 0, 1)
    skimage.io.imsave(filepath, deproc_img)


def comp_fet_mx(blob_data):
    fet_mx = np.reshape(
        blob_data[0],
        (blob_data.shape[1], blob_data.shape[2] * blob_data.shape[3])
    )
    return fet_mx


def comp_gram(blob_data):
    fet_mx = comp_fet_mx(blob_data)
    return np.dot(fet_mx, fet_mx.T)


def sgd(data, diff, solver_param):
    m = solver_param.get('m')  # momentum
    lr = solver_param.get('lr')  # learning rate

    if 'g_t' not in solver_param:
        solver_param['g_t'] = np.zeros_like(data)

    g_t = solver_param['g_t']
    g_t[:] = m * g_t - lr * diff

    data += g_t
    return g_t


def style_grad(gen_data, target_data):
    #print 'Stats:'
    #print np.std(gen_data)
    #print np.mean(gen_data)
    #print np.min(gen_data)
    #print np.max(gen_data)
    #print
    #pos_mask = gen_data > 0
    #non_z = target_data.copy()
    #non_z[~pos_mask] = 0
    #gram_mx_A = comp_gram(non_z)

    #non_z = gen_data.copy()
    #non_z[~pos_mask] = 0
    #gram_mx_G = comp_gram(non_z)

    gram_mx_A = comp_gram(target_data)
    gram_mx_G = comp_gram(gen_data)

    local_add = gram_mx_G - gram_mx_A
    fet_mx = comp_fet_mx(gen_data)
    size_sq = gen_data.size ** 2
    grad = np.dot(fet_mx.T, local_add).T
    grad = np.reshape(grad, gen_data.shape)
    #grad[~pos_mask] = 0
    #norm = np.abs(grad).mean() * size_sq
    norm = size_sq
    loss = np.sum(local_add ** 2) / 4

    grad /= norm
    loss /= norm

    return grad, loss


def content_grad(gen_data, target_data):
    #local_add[~pos_mask] = 0
    pos_mask = gen_data > 0
    grad = np.zeros_like(gen_data)
    grad[pos_mask] = gen_data[pos_mask] - target_data[pos_mask]
    #norm = np.abs(grad).mean()
    norm = 1
    loss = np.sum(grad ** 2) * 0.5

    grad /= norm
    loss /= norm

    return grad, loss


def gen_target_data(root_dir, caffe, net, get_data_blob, targets):
    target_data_list = []
    for target_i, (target_img_path, target_blob_names, _, _) in enumerate(targets):
        # Load and rescale to [0, 1]
        target_img = caffe.io.load_image(target_img_path)
        caffe_in = net.preprocess_inputs([target_img])
        # Copy image into input blob
        get_data_blob().data[...] = caffe_in
        net.forward()
        target_datas = {}
        for target_blob_name in target_blob_names:
            target_data = net.blobs[target_blob_name].data.copy()
            # Apply ReLU
            pos_mask = target_data > 0
            target_data[~pos_mask] = 0
            target_datas[target_blob_name] = target_data

        target_data_list.append(target_datas)

        save_image_blob(
            os.path.join(root_dir, 'target-{}.jpg'.format(target_i)),
            net,
            get_data_blob().data[0],
        )

    return target_data_list


def make_step(net, get_data_blob, all_target_blob_names, targets,
              target_data_list):
    # Makes one iteration step and updates the gradient of the data blob

    #with Timer('Forward'):
    net.forward(end=all_target_blob_names[-1])

    loss = 0
    # Go through target blobs in reversed order
    for i in range(len(all_target_blob_names)):
        blob_i = len(all_target_blob_names) - 1 - i
        start = all_target_blob_names[blob_i]

        if blob_i == 0:
            end = None
        else:
            end = all_target_blob_names[blob_i - 1]

        print 'Computing pass {} -> {}...'.format(start, end)
        # Get target blob
        target_blob = net.blobs[start]
        if i == 0:
            target_blob.diff[...] = 0

        gen_data = target_blob.data.copy()
        # Apply RELU
        pos_mask = gen_data > 0
        gen_data[~pos_mask] = 0

        # Go through all images and compute accumulated gradient for the current target blob
        #with Timer('Numpy magic'):
        target_blob_add_diff = np.zeros_like(target_blob.diff)
        for target_i, (_, target_blob_names, is_gram, weight) in enumerate(targets):
            # Skip if the current blob is not among the target's blobs
            if start not in target_blob_names:
                continue

            target_data = target_data_list[target_i][start]
            if is_gram:
                c_grad, c_loss = style_grad(gen_data, target_data)
            else:
                c_grad, c_loss = content_grad(gen_data, target_data)

            target_blob_add_diff += c_grad * weight / len(target_blob_names)
            loss += c_loss * weight / len(target_blob_names)

        target_blob.diff[...] += target_blob_add_diff

        #with Timer('Backward from {} to {}'.format(start, end)):
        net.backward(start=start, end=end)

    # normalize gradient
    get_data_blob().diff[...] /= np.abs(get_data_blob().diff).mean()

    return loss


def setup_classifier():
    deployfile_relpath = 'models/VGG_CNN_19/VGG_ILSVRC_19_layers_deploy_deepart.prototxt'
    weights_relpath = 'models/VGG_CNN_19/VGG_ILSVRC_19_layers.caffemodel'
    image_dims = (1014/2, 1280/2)
    #mean = (104, 117, 123)
    mean = (103.939, 116.779, 123.68)
    device_id = 0
    input_scale = 1.0

    caffe, net = load_caffe_net(
        deployfile_relpath, weights_relpath, image_dims, mean, device_id,
        input_scale
    )

    return caffe, net, image_dims


def deepart():
    np.random.seed(123)

    root_dir = 'gen_fet_image_debug'
    display = 100
    max_iter = 100000
    jitter = 0
    # list of targets defined by tuples of
    # (
    #     image path,
    #     target blob names (these activations will be included in the loss function),
    #     we use style (gram) or content loss,
    #     weighting factor
    # )
    targets = [
        ('images/starry_night.jpg', ['conv1_1'], True, 300),
        #('images/tuebingen.jpg'), ['conv4_2'], False, 1),
    ]
    # These have to be in the same order as in the network!
    #all_target_blob_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
    all_target_blob_names = ['conv1_1']
    #all_target_blob_names = ['conv4_2']

    caffe, net, image_dims = setup_classifier()

    ensuredir(root_dir)
    loss_filepath = os.path.join(root_dir, 'loss-fig.png')
    recent_loss_filepath = os.path.join(root_dir, 'loss-recent-fig.png')

    get_data_blob = lambda: net.blobs[net.inputs[0]]

    # Generate activations for input images
    target_data_list = gen_target_data(root_dir, caffe, net, get_data_blob, targets)

    # Modify one arbitrary activation
    #target_blob = target_data_list[0]['conv4_2']
    #channel = 1
    #target_blob[0, channel, ...] *= 5

    # Generate white noise image
    init_img = np.random.normal(loc=0.5, scale=0.001, size=image_dims + (3,))
    caffe_in = net.preprocess_inputs([init_img])
    # Copy image into input blob
    get_data_blob().data[...] = caffe_in

    loss_proc = []
    solver_param = {'lr': 0.01, 'm': 0.9}

    for it in range(max_iter):
        with Timer('Iteration'):
            if it % display == 0:
                print 'Saving image for iteration {}...'.format(it)
                # We have only one image as input
                save_image_blob(
                    os.path.join(root_dir, '{}-it.jpg'.format(it)),
                    net,
                    get_data_blob().data[0],
                )

                if it != 0:
                    save_loss_proc(recent_loss_filepath, loss_proc, 1000)
                    save_loss_proc(loss_filepath, loss_proc)

            # Add random shifting similar to google deep dream
            if jitter:
                ox, oy = np.random.randint(-jitter, jitter+1, 2)
                # apply jitter shift
                get_data_blob().data[0] = np.roll(
                    np.roll(get_data_blob().data[0], ox, -1), oy, -2
                )

            # The updated gradient will be saved to data_blob.diff
            loss = make_step(
                net, get_data_blob, all_target_blob_names, targets,
                target_data_list
            )
            loss_proc.append(loss)
            # SGD step
            sgd(get_data_blob().data, get_data_blob().diff, solver_param)

            if jitter:
                # unshift image
                get_data_blob().data[0] = np.roll(
                    np.roll(get_data_blob().data[0], -ox, -1), -oy, -2
                )



if __name__ == '__main__':
    deepart()
