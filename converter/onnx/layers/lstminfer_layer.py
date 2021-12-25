import logging
import numpy as np
from onnx import helper
from onnx import TensorProto as tp

from layers.base_layer import BaseLayer


class LSTMInferLayer(BaseLayer):
    def __init__(self, layer, name=None):
        super(LSTMInferLayer, self).__init__(layer, name)

    def get_lstm_attr(self, input_shape):

        attr_dict = {
            "hidden_size": [1],  # list of ints defaults is 1
        }

        attr_dict["hidden_size"] = self._layer.lstm_param.units
        attr_dict["layout"] = 0  # 1 cannot infer

        return attr_dict

    def create_lstm_input_weight(self, params: np.ndarray):
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

    def create_lstm_recurrent_weight(self, params: np.ndarray):
        param_name = self._layer.name + "_recurrent_weight"

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

    def create_lstm_input_bias(self, params: np.ndarray):
        param_name = self._layer.name + "_bias"

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

    def generate_node(self, shape):
        attr_dict = self.get_lstm_attr(shape)
        logging.debug(attr_dict)
        node = helper.make_node(
            "LSTM", self._in_names, self._out_names, self._layer.name, **attr_dict
        )
        logging.info("lstm_layer: " + self._layer.name + " created")
        self._node = node

    def generate_params(self, params):
        self.create_lstm_input_weight(params[0])
        if (len(params)) == 2:
            self.create_lstm_recurrent_weight(params[1])
        if (len(params)) == 3:
            a = np.zeros(4 * self._layer.lstm_param.units)
            c = np.concatenate((params[2], a))[None, ...]
            print(c.shape)
            self.create_lstm_recurrent_weight(params[1])
            self.create_lstm_input_bias(c)
