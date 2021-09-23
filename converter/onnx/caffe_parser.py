import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format


class caffe2onnx_converter():
    def __init__(self, proto_file, weight_file, onnx_file_name):
        self.proto_file = proto_file
        self.weight_file = weight_file
        self.onnx_file_name = onnx_file_name

    def run(self):
        self.net = self._load_caffe_net()
        self.model = self._load_caffe_weight()

        for layer in self.net.layer:
            pass

    def _load_caffe_net(self):
        net = caffe_pb2.NetParameter()
        with open(self.proto_file) as f:
            text_format.Merge(f.read(), net)
        
        return net

    def _load_caffe_weight(self):
        weight = caffe_pb2.NetParameter()
        with open(self.weight_file) as f:
            weight.ParseFromString(f.read())
        
        return weight