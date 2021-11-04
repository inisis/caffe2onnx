import logging
from onnx import helper

from layers.base_layer import BaseLayer


class ReluLayer(BaseLayer):
    def __init__(self, layer, name=None):
        super(ReluLayer, self).__init__(layer, name)

    def get_relu_attr(self):
        attr_dict = {"alpha": 0}
        if self._layer.type == "ReLU6":
            attr_dict["alpha"] = self._layer.relu6_param.negative_slope
        else:
            attr_dict["alpha"] = self._layer.relu_param.negative_slope

        return attr_dict

    def generate_node(self):
        attr_dict = self.get_relu_attr()
        node = helper.make_node(
            "LeakyRelu", self._in_names, self._out_names, self._layer.name, **attr_dict
        )
        logging.info("relu_layer: " + self._layer.name + " created")
        self._node = node
