import logging
from onnx import helper

from layers.base_layer import BaseLayer


class EluLayer(BaseLayer):
    def __init__(self, layer, name=None):
        super(EluLayer, self).__init__(layer, name)

    def get_relu_attr(self):
        attr_dict = {"alpha": 0}

        attr_dict["alpha"] = self._layer.elu_param.alpha

        return attr_dict

    def generate_node(self):
        attr_dict = self.get_relu_attr()
        node = helper.make_node(
            "Elu", self._in_names, self._out_names, self._layer.name, **attr_dict
        )
        logging.info("elu_layer: " + self._layer.name + " created")
        self._node = node
