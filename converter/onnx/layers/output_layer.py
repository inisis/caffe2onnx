import logging
from onnx import helper
from onnx import TensorProto as tp

from layers.base_layer import BaseLayer


class OutputLayer(BaseLayer):
    def __init__(self):
        super(OutputLayer, self).__init__()

    def _generate_output(self, output_name, shape):
        output_tvi = helper.make_tensor_value_info(output_name, tp.FLOAT, shape)
        logging.info("output_layer: " + output_name + " created")
        self._out_tensor_value_info.append(output_tvi)
