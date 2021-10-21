import logging
from onnx import helper
from onnx import TensorProto as tp

def get_conv_attr(layer):
    attr_dict = {
        "dilations": [1, 1], # list of ints defaults is 1
        "group": 1, # int default is 1
        "kernel_shape": 1, # list of ints If not present, should be inferred from input W.
        "pads": [0, 0, 0, 0], # list of ints defaults to 0
        "strides": [1, 1] # list of ints  defaults is 1
    }

    if layer.convolution_param.dilation != []:
        dilation = layer.convolution_param.dilation[0]
        dilations = [dilation, dilation]
        attr_dict["dilations"] = dilations
        
    if layer.convolution_param.pad != []:
        pads = list(layer.convolution_param.pad) * 4
        attr_dict["pads"] = pads
    elif layer.convolution_param.pad_h != 0 or layer.convolution_param.pad_w != 0:
        pads = [layer.convolution_param.pad_h, layer.convolution_param.pad_w] * 2
        attr_dict["pads"] = pads
    
    if layer.convolution_param.stride != []:
        strides = list(layer.convolution_param.stride) * 2
        attr_dict["strides"] = strides

    kernel_shape = list(layer.convolution_param.kernel_size) * 2
    if layer.convolution_param.kernel_size == []:
        kernel_shape = [layer.convolution_param.kernel_h, layer.convolution_param.kernel_w]
    
    attr_dict["kernel_shape"] = kernel_shape
    attr_dict["group"] = layer.convolution_param.group

    return attr_dict

def create_conv_weight(layer, params):
    param_name = layer.name + "_weight"

    param_type = tp.FLOAT
    param_shape = params.shape.dim
    param_tensor_value_info = helper.make_tensor_value_info(param_name, param_type, param_shape)
    param_tensor = helper.make_tensor(param_name, param_type, param_shape, params.data)

    return param_name, param_tensor_value_info, param_tensor

def create_conv_bias(layer, params):
    param_name = layer.name + "_bias"

    param_type = tp.FLOAT
    param_shape = params.shape.dim
    param_tensor_value_info = helper.make_tensor_value_info(param_name, param_type, param_shape)
    param_tensor = helper.make_tensor(param_name, param_type, param_shape, params.data)

    return param_name, param_tensor_value_info, param_tensor

def create_conv_layer(layer, node_name, input_name, output_name):
    attr_dict = get_conv_attr(layer)
    logging.debug(attr_dict)
    node = helper.make_node("Conv", input_name, output_name, node_name, **attr_dict)
    logging.info("conv_layer: " +node_name + " created")

    return node
