import logging
import numpy as np
from onnx import helper
from onnx import TensorProto as tp


from layers.base_layer import BaseLayer


class TileLayer(BaseLayer):
    def __init__(self, layer):
        super(TileLayer, self).__init__(layer)

    def create_tile_params(self, shape):
        attr_dict = {"axis": []}
        attr_dict["axis"] = self._layer.concat_param.axis

        length = len(shape)
        repeats = [1] * length
        idx = self._layer.tile_param.axis
        tile = self._layer.tile_param.tiles
        repeats[idx] = tile
        params = np.array(repeats)

        param_name = self._layer.name + "_tile"

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

    def generate_node(self, shape):
        self.create_tile_params(shape)

        node = helper.make_node(
            "Tile", self._in_names, self._out_names, self._layer.name
        )

        logging.info("tile_layer: " + self._layer.name + " created")
        self._node = node
