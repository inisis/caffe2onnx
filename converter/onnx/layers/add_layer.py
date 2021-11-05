import logging
import numpy as np
from onnx import helper
from onnx import TensorProto as tp

from layers.base_layer import BaseLayer


class AddLayer(BaseLayer):
    def __init__(self, layer, name=None):
        super(AddLayer, self).__init__(layer, name)

    def create_scale_bias(self, params: np.ndarray, shape):
        param_name = self._layer.name + "_weight"

        total_axes = len(shape)
        param_axes = len(params.shape)
        if (
            self._layer.scale_param.axis > -total_axes
            and self._layer.scale_param.axis <= -1
        ):
            axis = total_axes + self._layer.scale_param.axis
        elif (
            self._layer.scale_param.axis >= 0
            and self._layer.scale_param.axis < total_axes
        ):
            axis = self._layer.scale_param.axis
        else:
            raise Exception("unsupported axis: {}".format(self._layer.scale_param.axis))

        if self._layer.scale_param.num_axes == -1:
            shape_end = total_axes
            assert param_axes == (shape_end - axis)
        else:
            shape_end = axis + self._layer.scale_param.num_axes
            assert param_axes == (shape_end - axis)

        param_type = tp.FLOAT

        for idx in range(total_axes - axis - param_axes):
            params = params.reshape(*params.shape, 1)

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

    def create_log_bias(self, params: np.ndarray):
        param_name = self._layer.name + "_weight"

        param_type = tp.FLOAT

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
        node = helper.make_node(
            "Add", self._in_names, self._out_names, self._layer.name
        )
        logging.info("add_layer: " + self._layer.name + " created")
        self._node = node

    def generate_params(self, params, shape):
        if self._layer.type == "Scale":
            if (len(params)) == 2:
                self.create_scale_bias(params[1], shape)
        else:
            if (len(params)) == 2:
                self.create_log_bias(params[1])
