import logging
import numpy as np
from onnx import helper
from onnx import TensorProto as tp


from layers.base_layer import BaseLayer


class SqueezeLayer(BaseLayer):
    def __init__(self, layer, name=None):
        super(SqueezeLayer, self).__init__(layer, name)

    def create_squeeze_params(self, dim):
        axes = dim
        params = np.array(axes)

        param_name = self._layer.name + "_squeeze"

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

    def generate_node(self, dim):
        if self._layer.type == "LSTMInfer":
            self.create_squeeze_params(dim)

        node = helper.make_node(
            "Squeeze", self._in_names, self._out_names, self._layer.name
        )

        logging.info("squeeze_layer: " + self._layer.name + " created")
        self._node = node
