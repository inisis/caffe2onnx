import logging
import numpy as np
from onnx import helper
from onnx import TensorProto as tp

from layers.base_layer import BaseLayer


class DeconvLayer(BaseLayer):
    def __init__(self, layer, name=None):
        super(DeconvLayer, self).__init__(layer, name)

    def get_conv_attr(self):
        attr_dict = {
            "dilations": [1, 1],  # list of ints defaults is 1
            "group": 1,  # int default is 1
            "kernel_shape": 1,  # list of ints If not present, should be inferred from input W.
            "pads": [0, 0, 0, 0],  # list of ints defaults to 0
            "strides": [1, 1],  # list of ints  defaults is 1
        }

        if self._layer.convolution_param.dilation != []:
            dilation = self._layer.convolution_param.dilation[0]
            dilations = [dilation, dilation]
            attr_dict["dilations"] = dilations

        if self._layer.convolution_param.pad != []:
            pads = list(self._layer.convolution_param.pad) * 4
            attr_dict["pads"] = pads
        elif (
            self._layer.convolution_param.pad_h != 0
            or self._layer.convolution_param.pad_w != 0
        ):
            pads = [
                self._layer.convolution_param.pad_h,
                self._layer.convolution_param.pad_w,
            ] * 2
            attr_dict["pads"] = pads

        if self._layer.convolution_param.stride != []:
            strides = list(self._layer.convolution_param.stride) * 2
        elif (
            self._layer.convolution_param.stride_h != 0
            or self._layer.convolution_param.stride_w != 0
        ):
            strides = [
                self._layer.convolution_param.stride_h,
                self._layer.convolution_param.stride_w,
            ]
        else:
            strides = [1, 1]

        attr_dict["strides"] = strides

        kernel_shape = list(self._layer.convolution_param.kernel_size) * 2
        if self._layer.convolution_param.kernel_size == []:
            kernel_shape = [
                self._layer.convolution_param.kernel_h,
                self._layer.convolution_param.kernel_w,
            ]

        attr_dict["kernel_shape"] = kernel_shape
        attr_dict["group"] = self._layer.convolution_param.group

        return attr_dict

    def create_deconv_weight(self, params: np.ndarray):
        param_name = self._layer.name + "_weight"

        param_type = tp.FLOAT
        param_shape = params.shape
        param_tensor_value_info = helper.make_tensor_value_info(
            param_name, param_type, param_shape
        )
        param_tensor = helper.make_tensor(
            param_name, param_type, param_shape, params.flatten()
        )
        self._in_names.append(param_name)
        self._in_tensor_value_info.append(param_tensor_value_info)
        self._init_tensor.append(param_tensor)

    def create_deconv_bias(self, params: np.ndarray):
        param_name = self._layer.name + "_bias"

        param_type = tp.FLOAT
        param_shape = params.shape
        param_tensor_value_info = helper.make_tensor_value_info(
            param_name, param_type, param_shape
        )
        param_tensor = helper.make_tensor(
            param_name, param_type, param_shape, params.flatten()
        )
        self._in_names.append(param_name)
        self._in_tensor_value_info.append(param_tensor_value_info)
        self._init_tensor.append(param_tensor)

    def generate_node(self):
        attr_dict = self.get_conv_attr()
        logging.debug(attr_dict)
        node = helper.make_node(
            "ConvTranspose",
            self._in_names,
            self._out_names,
            self._layer.name,
            **attr_dict
        )
        logging.info("deconv_layer: " + self._layer.name + " created")
        self._node = node

    def generate_params(self, params):
        self.create_deconv_weight(params[0])
        if (len(params)) == 2:
            self.create_deconv_bias(params[1])
