import os

os.environ[
    "GLOG_minloglevel"
] = "3"  # 0 - debug 1 - info (still a LOT of outputs) 2 - warnings 3 - errors
import argparse
from converter.onnx.caffe_parser import caffe2onnx_converter

import pytest
import warnings

def test_mobilenet():
    prototxt = "test/caffemodel/mobilenet/mobilenet.prototxt"
    caffemodel = "test/caffemodel/mobilenet/mobilenet.caffemodel"
    converter = caffe2onnx_converter(prototxt, caffemodel, "test")
    converter.run()
    converter.test()

def test_mobilenetv2():
    prototxt = "test/caffemodel/mobilenetv2/mobilenet_v2.prototxt"
    caffemodel = "test/caffemodel/mobilenetv2/mobilenet_v2.caffemodel"
    converter = caffe2onnx_converter(prototxt, caffemodel, "test")
    converter.run()
    converter.test()

def test_resnet18():
    prototxt = "test/caffemodel/resnet18/resnet-18.prototxt"
    caffemodel = "test/caffemodel/resnet18/resnet-18.caffemodel"
    converter = caffe2onnx_converter(prototxt, caffemodel, "test")
    converter.run()
    converter.test()

def test_resnet50():
    prototxt = "test/caffemodel/resnet50/resnet-50-model.prototxt"
    caffemodel = "test/caffemodel/resnet50/resnet-50-model.caffemodel"
    converter = caffe2onnx_converter(prototxt, caffemodel, "test")
    converter.run()
    converter.test()

def test_yolov5():
    prototxt = "test/caffemodel/yolov5/yolov5s-4.0-not-focus-deconv.prototxt"
    caffemodel = "test/caffemodel/yolov5/yolov5s-4.0-not-focus-deconv.caffemodel"
    converter = caffe2onnx_converter(prototxt, caffemodel, "test")
    converter.run()
    converter.test()    
        
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test/caffemodel'])
