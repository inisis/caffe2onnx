import argparse

from converter.onnx.caffe_parser import caffe2onnx_converter

parser = argparse.ArgumentParser(description="Caffe 2 Onnx")
parser.add_argument(
    "caffe_proto_file", default=None, action="store", help="Caffe prototxt file"
)
parser.add_argument(
    "caffe_weight_file", default=None, action="store", help="Caffe weight file"
)
parser.add_argument(
    "onnx_file_name", default=None, action="store", help="Onnx file name for saved file"
)


def convert(args):
    converter = caffe2onnx_converter(
        args.caffe_proto_file, args.caffe_weight_file, args.onnx_file_name
    )
    converter.run()
    converter.test()
    converter.save()


def main():
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
