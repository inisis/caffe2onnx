import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

from onnx import save
from onnx import helper
from onnx import checker
from onnx import TensorProto as tp

import layers as ops

logging.basicConfig(format='%(asctime)s %(levelname)-5s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

class caffe2onnx_converter():
    def __init__(self, proto_file, weight_file, onnx_file_name):
        self.proto_file = proto_file
        self.weight_file = weight_file
        self.onnx_file_name = onnx_file_name
        self.save_path = os.path.dirname(os.path.abspath(self.proto_file))

    def run(self):
        self.net = self._load_caffe_net()
        self.model = self._load_caffe_weight()
        self.model_dict = self._generate_model_dict() # "Name": BLOBS
        self.in_tensor_value_info = []
        self.nodes = [] # nodes in graph
        self.input_names_set = set() # 
        self.out_tensor_value_info = []
        self.init_tensor = []

        for layer in self.net.layer:
            if layer.type == "Input":
                tensor_value_info = ops.create_input_layer(layer)
                self.in_tensor_value_info.append(tensor_value_info)
            elif layer.type == "Convolution":
                in_names = list(layer.bottom)
                self.input_names_set.update(in_names)
                out_names = list(layer.top)
                params = self.model_dict[layer.name]

                param_name, param_tensor_value_info, param_tensor = ops.create_conv_weight(layer, params[0])
                in_names.append(param_name)
                self.in_tensor_value_info.append(param_tensor_value_info)
                self.init_tensor.append(param_tensor)

                if(len(params)) == 2:
                    param_name, param_tensor_value_info, param_tensor = ops.create_conv_bias(layer, params[1])
                    in_names.append(param_name)
                    self.in_tensor_value_info.append(param_tensor_value_info)
                    self.init_tensor.append(param_tensor)

                conv_node = ops.create_conv_layer(layer, layer.name, in_names, out_names)

                self.nodes.append(conv_node)
            else:
                logging.error('unsupported layer type: %s', layer.type)

        self._get_output_tensor_value_info()

        graph_def = helper.make_graph(
            self.nodes,
            self.onnx_file_name,
            self.in_tensor_value_info,
            self.out_tensor_value_info,
            self.init_tensor
        )
        self.model_def = helper.make_model(graph_def, producer_name='caffe')
        checker.check_model(self.model_def)
        logging.info("onnx model conversion completed")

    def save(self):
        logging.info("onnx model saved to " + self.save_path + os.sep + self.onnx_file_name + ".onnx")
        save(self.model_def, self.save_path + os.sep + self.onnx_file_name + ".onnx")

    def _get_output_tensor_value_info(self):
        for node in self.nodes:
            if node.name not in self.input_names_set:
                self.out_tensor_value_info += ops.create_output_layer(node)


    def _load_caffe_net(self):
        net = caffe_pb2.NetParameter()
        with open(self.proto_file) as f:
            text_format.Merge(f.read(), net)
        
        return net

    def _load_caffe_weight(self):
        weight = caffe_pb2.NetParameter()
        with open(self.weight_file, 'rb') as f:
            weight.ParseFromString(f.read())
        
        return weight

    def _generate_model_dict(self):
        model_dict = {}
        for model_layer in self.model.layer:
            model_dict[model_layer.name] = model_layer.blobs
        
        return model_dict
