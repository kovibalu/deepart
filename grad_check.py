import numpy as np


def test_gradient_by_dim(objective_func, x0, dx, grad_err_thres, dim_idx, *args):
    x0_plus = x0.copy()
    x0_plus[dim_idx] += dx
    l_plus, _ = objective_func(x0_plus, *args)
    x0_minus = x0.copy()
    x0_minus[dim_idx] -= dx
    l_minus, _ = objective_func(x0_minus, *args)
    estimated_grad = float(l_plus - l_minus) / (2 * dx)

    loss, grad = objective_func(x0, *args)
    comp_grad = grad[dim_idx]
    scale = np.max([np.abs(estimated_grad), np.abs(comp_grad), 1.])

    grad_err = np.abs(estimated_grad - comp_grad)
    if grad_err > grad_err_thres * scale:
        print 'Finite difference gradient check error:', grad_err
        print 'Estimated gradient:', estimated_grad
        print 'Computed gradient:', comp_grad
        print 'Full computed gradient:', grad
        print 'Dimension index:', dim_idx
        return False

    return True


def gen_rand_idx(input_shape):
    dim_idxs = []
    for s in input_shape:
        dim_idx = np.random.randint(0, high=s)
        dim_idxs.append(dim_idx)

    return tuple(dim_idxs)


def test_gradient(objective_func, input_shape, dx, grad_err_thres, test_count, *args):
    success = True
    print 'Testing gradient...'
    for _ in range(test_count):
        print '.',
        dim_idx = gen_rand_idx(input_shape)
        # Testing with random input
        x0_rand = np.random.normal(size=input_shape)

        csuccess = test_gradient_by_dim(objective_func, x0_rand, dx, grad_err_thres, dim_idx, *args)
        if not csuccess:
            success = False

    print

    return success
