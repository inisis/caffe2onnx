import logging
from onnx import helper
from onnx import TensorProto as tp


from layers.base_layer import BaseLayer


class PoolingLayer(BaseLayer):
    def __init__(self, layer):
        super(PoolingLayer, self).__init__(layer)

    def get_pooling_attr(self, input_shape):
        attr_dict = {"kernel_shape": [], "strides": [1, 1], "pads": [0, 0, 0, 0]}
        if self._layer.pooling_param.kernel_size == 0:
            kernel_shape = [
                self._layer.pooling_param.kernel_h,
                self._layer.pooling_param.kernel_w,
            ]
        else:
            kernel_shape = [self._layer.pooling_param.kernel_size] * 2

        attr_dict["kernel_shape"] = kernel_shape
        if self._layer.pooling_param.stride != []:
            strides = [self._layer.pooling_param.stride] * 2

        attr_dict["strides"] = strides
        if self._layer.pooling_param.pad != 0:
            pads = [self._layer.pooling_param.pad] * 4
        elif (
            self._layer.pooling_param.pad_h != 0 or self._layer.pooling_param.pad_w != 0
        ):
            pads = [
                self._layer.pooling_param.pad_h,
                self._layer.pooling_param.pad_w,
            ] * 2
        else:
            pads = [0, 0, 0, 0]

        h = (input_shape[2] - kernel_shape[0] + pads[0] + pads[2]) / strides[0] + 1
        if h > int(h):
            pads[2] += 1

        w = (input_shape[3] - kernel_shape[1] + pads[1] + pads[3]) / strides[1] + 1
        if w > int(w):
            pads[3] += 1

        attr_dict["pads"] = pads

        return attr_dict

    def generate_node(self, input_shape):
        if self._layer.pooling_param.pool == 0:
            if self._layer.pooling_param.global_pooling == True:
                node = helper.make_node(
                    "GlobalMaxPool", self._in_names, self._out_names, self._layer.name
                )
            else:
                attr_dict = self.get_pooling_attr(input_shape)
                node = helper.make_node(
                    "MaxPool",
                    self._in_names,
                    self._out_names,
                    self._layer.name,
                    **attr_dict
                )

        elif self._layer.pooling_param.pool == 1:
            if self._layer.pooling_param.global_pooling == True:
                node = helper.make_node(
                    "GlobalAveragePool",
                    self._in_names,
                    self._out_names,
                    self._layer.name,
                )
            else:
                attr_dict = self.get_pooling_attr(input_shape)
                node = helper.make_node(
                    "AveragePool",
                    self._in_names,
                    self._out_names,
                    self._layer.name,
                    **attr_dict
                )

        logging.info("pooling_layer: " + self._layer.name + " created")
        self._node = node
