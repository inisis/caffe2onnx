import logging
import numpy as np
from onnx import helper
from onnx import TensorProto as tp


from layers.base_layer import BaseLayer


class PReluLayer(BaseLayer):
    def __init__(self, layer):
        super(PReluLayer, self).__init__(layer)

    def create_prelu_params(self, params, shape):
        param_name = self._layer.name + "_prelu"
        assert params.shape[0] == shape[1]

        param_type = tp.FLOAT
        param_shape = [1] * len(shape)
        param_shape[1] = params.shape[0]

        param_tensor_value_info = helper.make_tensor_value_info(
            param_name, param_type, param_shape
        )
        param_tensor = helper.make_tensor(
            param_name, param_type, param_shape, params.flatten()
        )
        self._in_names.append(param_name)
        self._in_tensor_value_info.append(param_tensor_value_info)
        self._init_tensor.append(param_tensor)

    def generate_node(self, params, shape):
        self.create_prelu_params(params, shape)

        node = helper.make_node(
            "PRelu", self._in_names, self._out_names, self._layer.name
        )

        logging.info("prelu_layer: " + self._layer.name + " created")
        self._node = node
