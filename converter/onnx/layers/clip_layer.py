import logging
from onnx import helper
from onnx import TensorProto as tp


from layers.base_layer import BaseLayer


class ClipLayer(BaseLayer):
    def __init__(self, layer, name=None):
        super(ClipLayer, self).__init__(layer, name)

    def create_clip_param(self, params, name):
        param_name = self._layer.name + name

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
            "Clip", self._in_names, self._out_names, (self._layer.name)
        )
        logging.info("clip_layer: " + self._layer.name + " created")
        self._node = node

    def generate_params(self, params):
        self.create_clip_param(params[0], "_min")
        self.create_clip_param(params[1], "_max")
