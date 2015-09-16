
import numpy as np

from grad_check import test_gradient
import deepart


def gen_test_target_data(net, targets):
    input_shape = deepart.get_data_blob(net).data.shape
    target_data_list = []
    for target_i, (target_img_path, target_blob_names, is_gram, _) in enumerate(targets):
        # Copy image into input blob
        target_data = np.random.normal(size=input_shape)
        deepart.get_data_blob(net).data[...] = target_data
        net.forward()
        target_datas = {}
        for target_blob_name in target_blob_names:
            target_data = net.blobs[target_blob_name].data.copy()
            # Apply ReLU
            pos_mask = target_data > 0
            target_data[~pos_mask] = 0
            if is_gram:
                target_datas[target_blob_name] = deepart.comp_gram(target_data)
            else:
                target_datas[target_blob_name] = target_data

        target_data_list.append(target_datas)

    return target_data_list


def test_all_gradients(init_img, net, all_target_blob_names, targets, target_data_list):
    # Set initial value and reshape net
    deepart.set_data(net, init_img)
    x0 = np.ravel(init_img).astype(np.float64)

    dx = 1e-2
    grad_err_thres = 1e-3
    test_count = 100

    input_shape = (1, 30, 40, 50)
    target_data = np.random.normal(size=input_shape)
    test_gradient(
        deepart.content_grad, input_shape, dx, grad_err_thres, test_count,
        target_data
    )
    target_data_gram = np.random.normal(size=(30, 30))
    test_gradient(
        deepart.style_grad, input_shape, dx, grad_err_thres, test_count,
        target_data_gram
    )

    target_data_list = gen_test_target_data(net, targets)
    test_gradient(
        deepart.objective_func, x0.shape, dx, grad_err_thres, test_count,
        net, all_target_blob_names, targets, target_data_list
    )

