import logging
from onnx import helper
from onnx import TensorProto as tp


from layers.base_layer import BaseLayer


class EltwiseLayer(BaseLayer):
    def __init__(self, layer, name=None):
        super(EltwiseLayer, self).__init__(layer, name)

    def generate_node(self):
        if self._layer.eltwise_param.operation == 0:
            node = helper.make_node(
                "Mul", self._in_names, self._out_names, self._layer.name
            )
        elif self._layer.eltwise_param.operation == 1:
            coeff = self._layer.eltwise_param.coeff
            if coeff[0] == 1 and coeff[1] == 1:
                node = helper.make_node(
                    "Add", self._in_names, self._out_names, self._layer.name
                )
            if coeff[0] == 1 and coeff[1] == -1:
                node = helper.make_node(
                    "Sub", self._in_names, self._out_names, self._layer.name
                )
            if coeff[0] == -1 and coeff[1] == -1:
                node = helper.make_node(
                    "Add", self._in_names, self._out_names, self._layer.name
                )
            if coeff[0] == -1 and coeff[1] == 1:
                node = helper.make_node(
                    "Sub", self._in_names, self._out_names, self._layer.name
                )
        elif self._layer.eltwise_param.operation == 2:
            node = helper.make_node(
                "Max", self._in_names, self._out_names, self._layer.name
            )

        logging.info("eltwise_layer: " + self._layer.name + " created")
        self._node = node
