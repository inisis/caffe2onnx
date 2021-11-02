import logging
import numpy as np
from onnx import helper
from onnx import TensorProto as tp

from layers.base_layer import BaseLayer


class ScaleLayer(BaseLayer):
    def __init__(self, layer):
        super(ScaleLayer, self).__init__(layer)

    def get_scale_attr(self):
        attr_dict = {
            "axis ": 1,  # int defaults is 1
            "num_axes ": 1,  # int default is 1
        }

        attr_dict["axis"] = self._layer.scale_param.axis
        attr_dict["num_axes"] = self._layer.scale_param.num_axes

        return attr_dict

    def create_scale_weight(self, params: np.ndarray):
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

    def create_scale_bias(self, params: np.ndarray):
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
            "Mul", self._in_names, self._out_names, self._layer.name, **attr_dict
        )
        logging.info("scale_layer: " + self._layer.name + " created")
        self._node = node

    def generate_params(self, params):
        self.create_scale_weight(params[0])
        if (len(params)) == 2:
            self.create_scale_bias(params[1])
