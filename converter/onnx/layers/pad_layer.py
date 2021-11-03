import logging
import numpy as np
from onnx import helper
from onnx import TensorProto as tp


from layers.base_layer import BaseLayer


class PadLayer(BaseLayer):
    def __init__(self, layer, name=None):
        super(PadLayer, self).__init__(layer, name)

    def create_pad_params(self):
        pad = self._layer.pooling_param.pad
        if pad != 0:
            pad_h = pad_w = pad
        else:
            if self._layer.pooling_param.pad_h != 0 and self._layer.pooling_param.pad_w != 0:
                pad_h = self._layer.pooling_param.pad_h
                pad_w = self._layer.pooling_param.pad_w
            else:
                pad_h = pad_w = 0
        pads = [0, 0, pad_h, pad_w, 0, 0, pad_h, pad_w]

        params = np.array(pads)

        param_name = self._layer.name + "_pad"

        param_type = tp.INT64
        param_shape = params.shape
        param_tensor_value_info = helper.make_tensor_value_info(
            param_name, param_type, param_shape
        )
        param_tensor = helper.make_tensor(
            param_name, param_type, param_shape, params.flatten()
        )
        self._in_names.append(param_name)
        self._in_tensor_value_info.append(param_tensor_value_info)
        self._init_tensor.append(param_tensor)

    def generate_node(self):
        if self._layer.pooling_param.pool == 1 and self._layer.pooling_param.global_pooling != True:
            self.create_pad_params()

        node = helper.make_node(
            "Pad", self._in_names, self._out_names, self._layer.name, mode='constant'
        )

        logging.info("pad_layer: " + self._layer.name + " created")
        self._node = node
