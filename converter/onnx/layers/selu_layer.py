import logging
import numpy as np
from onnx import helper
from onnx import TensorProto as tp

from layers.base_layer import BaseLayer


class SeluLayer(BaseLayer):
    def __init__(self, layer):
        super(SeluLayer, self).__init__(layer)

    def get_selu_attr(self):
        attr_dict = {
            "alpha": 1.67326319217681884765625,  # float defaults is 1.67326319217681884765625
            "gamma": 1.05070102214813232421875,  # float default is 1.05070102214813232421875
        }

        attr_dict["alpha"] = self._layer.elu_param.alpha
        attr_dict["gamma"] = self._layer.elu_param.beta

        return attr_dict

    def generate_node(self):
        attr_dict = self.get_selu_attr()
        logging.debug(attr_dict)
        node = helper.make_node(
            "Selu", self._in_names, self._out_names, self._layer.name, **attr_dict
        )
        logging.info("selu_layer: " + self._layer.name + " created")
        self._node = node
