import logging
from onnx import helper
from onnx import TensorProto as tp

def create_output_layer(node):
    output_tvi_list = []
    output_tvi = helper.make_tensor_value_info(node.name, tp.FLOAT, [3])
    output_tvi_list.append(output_tvi)
    logging.info("output_layer: " + node.name + " created")

    return output_tvi_list