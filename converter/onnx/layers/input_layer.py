import logging
from onnx import helper
from onnx import TensorProto as tp

from layers.base_layer import BaseLayer


class InputLayer(BaseLayer):
    def __init__(self):
        super(InputLayer, self).__init__()

    def _generate_input(self, input_name, shape):
        input_tvi = helper.make_tensor_value_info(input_name, tp.FLOAT, shape)
        logging.info("input_layer: " + input_name + " created")
        self._in_tensor_value_info.append(input_tvi)
