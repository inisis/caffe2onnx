import logging
import numpy as np
from onnx import helper


from layers.base_layer import BaseLayer


class ReductionLayer(BaseLayer):
    def __init__(self, layer, name=None):
        super(ReductionLayer, self).__init__(layer, name)

    def generate_node(self, shape):
        axis = self._layer.reduction_param.axis
        if axis == len(shape):
            axes = [axis]
        else:
            axes = np.arange(axis, len(shape)).tolist()

        if self._layer.reduction_param.operation == 1:
            node = helper.make_node(
                "ReduceSum",
                self._in_names,
                self._out_names,
                self._layer.name,
                keepdims=0,
                axes=axes,
            )
        elif self._layer.reduction_param.operation == 2:
            node = helper.make_node(
                "ReduceSum",
                self._in_names,
                self._out_names,
                self._layer.name,
                keepdims=0,
                axes=axes,
            )
        elif self._layer.reduction_param.operation == 3:
            node = helper.make_node(
                "ReduceSumSquare",
                self._in_names,
                self._out_names,
                self._layer.name,
                keepdims=0,
                axes=axes,
            )
        elif self._layer.reduction_param.operation == 4:
            node = helper.make_node(
                "ReduceMean",
                self._in_names,
                self._out_names,
                self._layer.name,
                keepdims=0,
                axes=axes,
            )

        logging.info("eltwise_layer: " + self._layer.name + " created")
        self._node = node
