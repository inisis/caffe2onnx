import logging
from onnx import helper
from onnx import TensorProto as tp


from layers.base_layer import BaseLayer


class SqrtLayer(BaseLayer):
    def __init__(self, layer):
        super(SqrtLayer, self).__init__(layer)

    def generate_node(self):
        node = helper.make_node(
            "Sqrt", self._in_names, self._out_names, self._layer.name
        )
        logging.info("sqrt_layer: " + self._layer.name + " created")
        self._node = node
