import logging
import numpy as np
from onnx import helper
from onnx import TensorProto as tp

from layers.base_layer import BaseLayer


class PermuteLayer(BaseLayer):
    def __init__(self, layer, name=None):
        super(PermuteLayer, self).__init__(layer, name)

    def get_permute_attr(self, shape):
        attr_dict = {"perm": []}
        if self._layer.type == "Permute":
            if len(self._layer.permute_param.order) != len(shape):
                order_list = np.arange(len(shape)).tolist()
                permute_order = self._layer.permute_param.order
                for element in permute_order:
                    order_list.remove(element)
                permute_order.extend(order_list)
                attr_dict["perm"] = permute_order
            else:
                attr_dict["perm"] = self._layer.permute_param.order
        elif self._layer.type == "Softmax":
            axis = self._layer.softmax_param.axis
            index = np.arange(len(shape))
            index[axis], index[-1] = index[-1], index[axis]
            attr_dict["perm"] = index

        return attr_dict

    def get_softmax_permute_attr(self, params=None):
        attr_dict = {"perm": []}
        attr_dict["perm"] = params

        return attr_dict

    def generate_node(self, shape):
        attr_dict = self.get_permute_attr(shape)
        node = helper.make_node(
            "Transpose", self._in_names, self._out_names, self._layer.name, **attr_dict
        )

        logging.info("permute_layer: " + self._layer.name + " created")
        self._node = node
