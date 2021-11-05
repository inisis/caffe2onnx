import logging
import numpy as np
from onnx import helper
from onnx import TensorProto as tp


from layers.base_layer import BaseLayer


class PowerLayer(BaseLayer):
    def __init__(self, layer, name=None):
        super(PowerLayer, self).__init__(layer, name)

    def create_power_weight(self, params: np.ndarray):
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

    def generate_node(self):
        node = helper.make_node(
            "Pow", self._in_names, self._out_names, self._layer.name
        )
        logging.info("power_layer: " + self._layer.name + " created")
        self._node = node

    def generate_params(self, params):
        self.create_power_weight(params)
