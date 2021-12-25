import logging
from onnx import helper

from layers.base_layer import BaseLayer


class TransposeLayer(BaseLayer):
    def __init__(self, layer, name=None):
        super(TransposeLayer, self).__init__(layer, name)

    def gen_transpose_attr(self, perm):
        attr_dict = {"perm": perm}

        return attr_dict

    def generate_node(self, perm):
        attr_dict = self.gen_transpose_attr(perm)
        node = helper.make_node(
            "Transpose", self._in_names, self._out_names, self._layer.name, **attr_dict
        )
        logging.info("transpose_layer: " + self._layer.name + " created")
        self._node = node
