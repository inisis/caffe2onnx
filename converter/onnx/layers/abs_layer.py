import logging
from onnx import helper

from layers.base_layer import BaseLayer


class AbsLayer(BaseLayer):
    def __init__(self, layer, name=None):
        super(AbsLayer, self).__init__(layer, name)

    def generate_node(self):
        node = helper.make_node(
            "Abs", self._in_names, self._out_names, self._layer.name
        )
        logging.info("abs_layer: " + self._layer.name + " created")
        self._node = node
