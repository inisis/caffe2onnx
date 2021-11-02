import copy
import logging
from onnx import helper
from onnx import TensorProto as tp


class BaseLayer(object):
    def __init__(self, layer=None, name=None):
        self._in_tensor_value_info = []
        self._init_tensor = []
        self._out_tensor_value_info = []
        self._node = None

        self._in_names = []
        self._out_names = []
        self._layer = copy.deepcopy(layer)
        if layer != None and name != None:
            self._layer.name += name
        self._is_inplace = False

        self._check_is_inplace()

    def generate_node(self):
        pass

    def generate_params(self, params):
        pass

    def _check_is_inplace(self):
        if self._layer is None:
            return
        elif self._layer.top[0] == self._layer.bottom[0]:
            self._is_inplace = True
