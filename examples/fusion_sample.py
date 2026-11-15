"""Synthetic illustrative code, NOT PyTorch/XLA source and NOT a verified optimization."""
def example_fuse_conv_bn(conv, bn, training=False):
    if not training and conv.out_channels == bn.num_features:
        return "fold_bn_into_conv"
    return "separate_conv_and_bn"

def example_epilogue(matmul, bias, relu, enable_epilogue=True):
    if enable_epilogue and bias is not None:
        return "matmul_bias_relu_epilogue"
    return "separate_kernels"
