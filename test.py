import os

os.environ[
    "GLOG_minloglevel"
] = "3"  # 0 - debug 1 - info (still a LOT of outputs) 2 - warnings 3 - errors
import glob
import argparse
from converter.onnx.caffe_parser import caffe2onnx_converter

parser = argparse.ArgumentParser(description="Auto test")
parser.add_argument(
    "--base_dir", default=None, type=str, help="Base dir to generate csv"
)


def find_all_files(base):
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.endswith(".prototxt"):
                fullname = os.path.join(root, f)
                yield fullname


def run(args):
    with open("test_result.txt", "w") as f:
        lsts = os.listdir(args.base_dir)
        lsts.sort()
        for lst in lsts:
            sub_dir = args.base_dir + os.sep + lst
            for prototxt in find_all_files(sub_dir):
                caffemodel = prototxt.replace("prototxt", "caffemodel")
                print(caffemodel)
                print(prototxt)
                case = os.path.basename(os.path.dirname(caffemodel))
                converter = caffe2onnx_converter(prototxt, caffemodel, "test")
                converter.run()
                try:
                    converter.test()
                except:
                    print("case: " + case + " fail")
                    f.write("case: " + case + " fail" + "\n")
                    continue
                converter.save()

                print("case: " + case + " pass")
                f.write("case: " + case + " pass" + "\n")


def main():
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
