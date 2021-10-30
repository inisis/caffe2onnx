import logging
import numpy as np
from onnx import helper
from onnx import TensorProto as tp

from layers.base_layer import BaseLayer


class GemmLayer(BaseLayer):
    def __init__(self, layer):
        super(GemmLayer, self).__init__(layer)

    def get_gemm_attr(self):
        attr_dict = {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 1}

        return attr_dict

    def create_gemm_weight(self, params: np.ndarray):
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

    def create_gemm_bias(self, params: np.ndarray):
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
        attr_dict = self.get_gemm_attr()
        logging.debug(attr_dict)
        node = helper.make_node(
            "Gemm", self._in_names, self._out_names, self._layer.name, **attr_dict
        )
        logging.info("gemm_layer: " + self._layer.name + " created")
        self._node = node

    def generate_params(self, params):
        self._layer.name = self._layer.name + "_gemm"
        self.create_gemm_weight(params[0])
        if (len(params)) == 2:
            self.create_gemm_bias(params[1])
