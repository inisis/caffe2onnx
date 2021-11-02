import logging
from onnx import helper
from onnx import TensorProto as tp


from layers.base_layer import BaseLayer


class PermuteLayer(BaseLayer):
    def __init__(self, layer):
        super(PermuteLayer, self).__init__(layer)

    def get_permute_attr(self):
        attr_dict = {"perm": []}
        attr_dict["perm"] = self._layer.permute_param.order

        return attr_dict

    def generate_node(self):
        attr_dict = self.get_permute_attr()
        node = helper.make_node(
            "Transpose", self._in_names, self._out_names, self._layer.name, **attr_dict
        )

        logging.info("permute_layer: " + self._layer.name + " created")
        self._node = node
