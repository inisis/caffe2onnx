import logging
from onnx import helper
from onnx import TensorProto as tp

def create_input_layer(layer):
    node = helper.make_tensor_value_info(layer.name, tp.FLOAT, layer.input_param.shape[0].dim)
    logging.info("input_layer: " + layer.name + " created")

    return node