import logging
import numpy as np
from onnx import helper
from onnx import TensorProto as tp


from layers.base_layer import BaseLayer


class Reshapelayer(BaseLayer):
    def __init__(self, layer):
        super(Reshapelayer, self).__init__(layer)
    
    def create_reshape_params_inner_product(self, params):
        param_name = self._layer.name + "_shape"

        params_new = np.array([params[0], np.prod(params[1:])])

        param_type = tp.INT64
        param_shape = params_new.shape

        param_tensor_value_info = helper.make_tensor_value_info(
            param_name, param_type, param_shape
        )
        param_tensor = helper.make_tensor(
            param_name, param_type, param_shape, params_new.flatten()
        )

        self._in_names.append(param_name)
        self._in_tensor_value_info.append(param_tensor_value_info)
        self._init_tensor.append(param_tensor)

    def create_reshape_params(self):
        param_name = self._layer.name + "_shape"

        params = np.array(self._layer.reshape_param.shape.dim)

        param_type = tp.INT64
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
        node = helper.make_node(
            "Reshape", self._in_names, self._out_names, self._layer.name
        )

        logging.info("reshape_layer: " + self._layer.name + " created")
        self._node = node

    def generate_params(self, params=None):
        if params is not None:
            self._layer.name = self._layer.name + "_reshape"
            self.create_reshape_params_inner_product(params)
        else:
            self.create_reshape_params()

