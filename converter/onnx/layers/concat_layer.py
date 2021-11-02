import logging
from onnx import helper
from onnx import TensorProto as tp


from layers.base_layer import BaseLayer


class ConcatLayer(BaseLayer):
    def __init__(self, layer):
        super(ConcatLayer, self).__init__(layer)

    def get_concat_attr(self):
        attr_dict = {"axis": []}
        attr_dict["axis"] = self._layer.concat_param.axis

        return attr_dict

    def generate_node(self, axis=None):
        if axis == None:
            attr_dict = self.get_concat_attr()
        else:
            attr_dict = {"axis": axis}

        node = helper.make_node(
            "Concat", self._in_names, self._out_names, self._layer.name, **attr_dict
        )

        logging.info("concat_layer: " + self._layer.name + " created")
        self._node = node
