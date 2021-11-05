import logging
from onnx import helper
from onnx import TensorProto as tp


from layers.base_layer import BaseLayer


class CosineLayer(BaseLayer):
    def __init__(self, layer, name=None):
        super(CosineLayer, self).__init__(layer, name)

    def generate_node(self):
        node = helper.make_node(
            "Cos", self._in_names, self._out_names, self._layer.name
        )
        logging.info("cos_layer: " + self._layer.name + " created")
        self._node = node
