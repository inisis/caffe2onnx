import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

import numpy as np

from onnx import save, helper, checker

import onnxruntime as rt

import layers as ops

logging.basicConfig(
    format="%(asctime)s %(levelname)-5s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class caffe2onnx_converter:
    def __init__(self, proto_file, weight_file, onnx_file_name):
        self.proto_file = proto_file
        self.weight_file = weight_file
        self.onnx_file_name = onnx_file_name
        self.save_path = os.path.dirname(os.path.abspath(self.proto_file))

    def run(self):
        self.model_def = self._load_caffe_prototxt()
        self.model_weights = self._load_caffe_weight()
        self.caffe_net = self._load_net()

        self.in_tensor_value_info = []
        self.nodes = []  # nodes in graph
        self.out_tensor_value_info = []
        self.init_tensor = []
        self.inplace_dict = {}

        for layer in self.model_def.layer:
            if layer.type == "Input":
                pass
            elif layer.type == "Convolution" or layer.type == "DepthwiseConvolution":
                conv_layer = ops.ConvLayer(layer)
                for idx in range(len(layer.bottom)):
                    if layer.bottom[idx] in self.inplace_dict.keys():
                        last_key = list(self.inplace_dict[layer.bottom[idx]].keys())[-1]
                        last_layer_output_name = self.inplace_dict[layer.bottom[idx]][
                            last_key
                        ]["new_output"]
                        conv_layer._in_names.append(last_layer_output_name)
                    else:
                        conv_layer._in_names.extend(list(layer.bottom))

                conv_layer._out_names.extend(list(layer.top))

                params = self.caffe_net.params[layer.name]
                params_numpy = self._param_to_numpy(params)
                shape = self.caffe_net.blobs[layer.bottom[0]].data.shape
                conv_layer.generate_params(params_numpy)
                conv_layer.generate_node(shape)

                self._node_post_process(conv_layer)
            elif layer.type == "BatchNorm":
                batchnorm_layer = ops.BatchNormLayer(layer)

                batchnorm_layer._in_names.extend(list(layer.bottom))

                if batchnorm_layer._is_inplace == True:
                    this_layer_output_name = layer.name + "_output"
                    self._update_inplace_dict(
                        layer, layer.top[0], this_layer_output_name
                    )
                    batchnorm_layer._out_names.append(this_layer_output_name)
                else:
                    batchnorm_layer._out_names.extend(list(layer.top))

                params_batchnorm = self.caffe_net.params[layer.name]
                params_batchnorm_numpy = self._param_to_numpy(params_batchnorm)

                idx = self._get_layer_index(layer)
                if (
                    idx < self._get_net_length() - 1
                    and self.caffe_net.layers[idx + 1].type == "Scale"
                ):
                    params_scale = self.caffe_net.params[
                        self.caffe_net._layer_names[idx + 1]
                    ]
                    params_scale_numpy = self._param_to_numpy(params_scale)
                    batchnorm_layer.generate_params(
                        params_batchnorm_numpy, params_scale_numpy
                    )
                else:
                    default_param_scale = [
                        np.ones(shape=params_batchnorm_numpy[0].shape, dtype=np.float),
                        np.zeros(shape=params_batchnorm_numpy[1].shape, dtype=np.float),
                    ]
                    batchnorm_layer.generate_params(
                        params_batchnorm_numpy, default_param_scale
                    )

                batchnorm_layer.generate_node()

                self._node_post_process(batchnorm_layer)
            elif layer.type == "Scale":
                idx = self._get_layer_index(layer)
                if self.caffe_net.layers[idx - 1].type == "BatchNorm":
                    continue
                else:
                    # scale = Mul + Add
                    mul_layer = ops.MulLayer(layer, "_mul")

                    params_scale = self.caffe_net.params[layer.name]
                    params_scale_numpy = self._param_to_numpy(params_scale)
                    shape = self.caffe_net.blobs[layer.bottom[0]].data.shape
                    if len(params_scale_numpy) == 2:
                        mul_out_name = layer.name + "_mul_out"

                        mul_layer._in_names.extend(list(layer.bottom))
                        mul_layer._out_names.append(mul_out_name)

                        mul_layer.generate_params(params_scale_numpy, shape)
                        mul_layer.generate_node()

                        self._node_post_process(mul_layer)

                        add_layer = ops.AddLayer(layer, "_add")

                        add_layer._in_names.append(mul_out_name)
                        add_layer._out_names.extend(list(layer.top))

                        add_layer.generate_params(params_scale_numpy, shape)
                        add_layer.generate_node()

                        self._node_post_process(add_layer)
                    else:
                        mul_layer._in_names.extend(list(layer.bottom))
                        mul_layer._out_names.extend(list(layer.top))

                        mul_layer.generate_params(params_scale_numpy, shape)
                        mul_layer.generate_node()

                        self._node_post_process(mul_layer)
            elif layer.type == "ReLU":
                relu_layer = ops.ReluLayer(layer)
                if relu_layer._is_inplace == True:
                    this_layer_output_name = layer.name + "_output"
                    if layer.top[0] in self.inplace_dict.keys():
                        last_key = list(self.inplace_dict[layer.top[0]].keys())[-1]
                        last_layer_output_name = self.inplace_dict[layer.top[0]][
                            last_key
                        ]["new_output"]
                        relu_layer._in_names.append(last_layer_output_name)
                        self._update_inplace_dict(
                            layer, last_layer_output_name, this_layer_output_name
                        )
                    else:
                        relu_layer._in_names.append(layer.top[0])
                        self._update_inplace_dict(
                            layer, layer.top[0], this_layer_output_name
                        )

                    relu_layer._out_names.append(this_layer_output_name)
                else:
                    relu_layer._in_names.extend(list(layer.bottom))
                    relu_layer._out_names.extend(list(layer.top))

                relu_layer.generate_node()

                self._node_post_process(relu_layer)
            elif layer.type == "Pooling":
                if (
                    layer.pooling_param.pool == 1
                    and layer.pooling_param.global_pooling != True
                ):
                    for idx in range(len(layer.bottom)):
                        if layer.bottom[idx] in self.inplace_dict.keys():
                            last_key = list(
                                self.inplace_dict[layer.bottom[idx]].keys()
                            )[-1]
                            last_layer_output_name = self.inplace_dict[
                                layer.bottom[idx]
                            ][last_key]["new_output"]
                        else:
                            last_layer_output_name = layer.bottom[idx]
                    pad_layer = ops.PadLayer(layer, "_pad")
                    pad_layer_out_name = layer.name + "_pad_out"
                    pad_layer._in_names.append(last_layer_output_name)
                    pad_layer._out_names.append(pad_layer_out_name)
                    pad_layer.generate_node()

                    self._node_post_process(pad_layer)

                    pooling_layer = ops.PoolingLayer(layer)
                    pooling_layer._in_names.append(pad_layer_out_name)
                    pooling_layer._out_names.extend(list(layer.top))
                    shape = self.caffe_net.blobs[layer.bottom[0]].data.shape

                    pooling_layer.generate_node(shape)
                    self._node_post_process(pooling_layer)
                else:
                    pooling_layer = ops.PoolingLayer(layer)
                    for idx in range(len(layer.bottom)):
                        if layer.bottom[idx] in self.inplace_dict.keys():
                            last_key = list(
                                self.inplace_dict[layer.bottom[idx]].keys()
                            )[-1]
                            last_layer_output_name = self.inplace_dict[
                                layer.bottom[idx]
                            ][last_key]["new_output"]
                        else:
                            last_layer_output_name = layer.bottom[idx]

                    pooling_layer._in_names.append(last_layer_output_name)
                    pooling_layer._out_names.extend(list(layer.top))

                    shape = self.caffe_net.blobs[layer.bottom[0]].data.shape

                    pooling_layer.generate_node(shape)
                    self._node_post_process(pooling_layer)
            elif layer.type == "Eltwise":
                eltwise_layer = ops.EltwiseLayer(layer)

                for idx in range(len(layer.bottom)):
                    if layer.bottom[idx] in self.inplace_dict.keys():
                        last_key = list(self.inplace_dict[layer.bottom[idx]].keys())[-1]
                        last_layer_output_name = self.inplace_dict[layer.bottom[idx]][
                            last_key
                        ]["new_output"]
                        eltwise_layer._in_names.append(last_layer_output_name)
                    else:
                        eltwise_layer._in_names.append(layer.bottom[idx])
                if (
                    len(list(layer.eltwise_param.coeff)) != 0
                    and list(layer.eltwise_param.coeff)[0] == -1
                ):
                    eltwise_out_name = layer.name + "_eltwist_out"
                    eltwise_layer._out_names.append(eltwise_out_name)
                    eltwise_layer.generate_node()

                    self._node_post_process(eltwise_layer)

                    neg_layer = ops.NegLayer(layer, "_neg")
                    neg_output_name = layer.name + "_neg_out"
                    neg_layer._in_names.append(eltwise_out_name)
                    neg_layer._out_names.extend(list(layer.top))
                    neg_layer.generate_node()

                    self._node_post_process(neg_layer)
                else:
                    eltwise_layer._out_names.extend(list(layer.top))
                    eltwise_layer.generate_node()

                    self._node_post_process(eltwise_layer)
            elif layer.type == "InnerProduct":
                reshape_layer = ops.Reshapelayer(layer, "_reshape")

                shape = self.caffe_net.blobs[layer.bottom[0]].data.shape
                reshape_out_name = layer.name + "_reshape_out"

                reshape_layer._in_names.extend(list(layer.bottom))
                reshape_layer._out_names.append(reshape_out_name)

                reshape_layer.generate_params(shape)
                reshape_layer.generate_node()

                self._node_post_process(reshape_layer)

                gemm_layer = ops.GemmLayer(layer, "_gemm")
                gemm_layer._in_names.append(reshape_out_name)
                gemm_layer._out_names.extend(list(layer.top))
                params = self.caffe_net.params[layer.name]
                params_numpy = self._param_to_numpy(params)

                gemm_layer.generate_params(params_numpy)
                gemm_layer.generate_node()

                self._node_post_process(gemm_layer)
            elif layer.type == "Softmax":
                permute_layer = ops.PermuteLayer(layer, "_permute")
                shape = self.caffe_net.blobs[layer.bottom[0]].data.shape
                permute_out_name = layer.name + "_permute_out"
                permute_layer._in_names.extend(list(layer.bottom))
                permute_layer._out_names.append(permute_out_name)
                permute_layer.generate_node(shape)

                self._node_post_process(permute_layer)

                softmax_layer = ops.SoftmaxLayer(layer, "_softmax")
                softmax_out_name = layer.name + "_softmax_out"
                softmax_layer._in_names.append(permute_out_name)
                softmax_layer._out_names.append(softmax_out_name)
                softmax_layer.generate_node()

                self._node_post_process(softmax_layer)

                permute_layer = ops.PermuteLayer(layer, "_post_permute")
                shape = self.caffe_net.blobs[layer.bottom[0]].data.shape
                permute_out_name = layer.name + "_permute_out"
                permute_layer._in_names.append(softmax_out_name)
                permute_layer._out_names.extend(list(layer.top))
                permute_layer.generate_node(shape)

                self._node_post_process(permute_layer)

            elif layer.type == "Sigmoid":
                sigmoid_layer = ops.SigmoidLayer(layer)
                sigmoid_layer._in_names.extend(list(layer.bottom))
                sigmoid_layer._out_names.extend(list(layer.top))
                sigmoid_layer.generate_node()

                self._node_post_process(sigmoid_layer)
            elif layer.type == "Concat":
                concat_layer = ops.ConcatLayer(layer)
                concat_layer._in_names.extend(list(layer.bottom))
                concat_layer._out_names.extend(list(layer.top))
                concat_layer.generate_node()

                self._node_post_process(concat_layer)
            elif layer.type == "Deconvolution":
                deconv_layer = ops.DeconvLayer(layer)
                for idx in range(len(layer.bottom)):
                    if layer.bottom[idx] in self.inplace_dict.keys():
                        last_key = list(self.inplace_dict[layer.bottom[idx]].keys())[-1]
                        last_layer_output_name = self.inplace_dict[layer.bottom[idx]][
                            last_key
                        ]["new_output"]
                        deconv_layer._in_names.append(last_layer_output_name)
                    else:
                        deconv_layer._in_names.extend(list(layer.bottom))

                deconv_layer._out_names.extend(list(layer.top))

                params = self.caffe_net.params[layer.name]
                params_numpy = self._param_to_numpy(params)

                deconv_layer.generate_params(params_numpy)
                deconv_layer.generate_node()

                self._node_post_process(deconv_layer)
            elif layer.type == "Reshape":
                reshape_layer = ops.Reshapelayer(layer)

                reshape_layer._in_names.extend(list(layer.bottom))
                reshape_layer._out_names.extend(list(layer.top))

                shape = self.caffe_net.blobs[layer.bottom[0]].data.shape

                reshape_layer.generate_params(shape=shape)
                reshape_layer.generate_node()

                self._node_post_process(reshape_layer)
            elif layer.type == "Permute":
                permute_layer = ops.PermuteLayer(layer)

                permute_layer._in_names.extend(list(layer.bottom))
                permute_layer._out_names.extend(list(layer.top))

                shape = self.caffe_net.blobs[layer.bottom[0]].data.shape
                permute_layer.generate_node(shape)

                self._node_post_process(permute_layer)
            elif layer.type == "Log":
                # log layer = log (mul + add)
                assert layer.log_param.base == -1  # log base e
                mul_layer = ops.MulLayer(layer, "_mul")
                mul_out_name = layer.name + "_mul_out"

                mul_layer._in_names.extend(list(layer.bottom))
                mul_layer._out_names.append(mul_out_name)

                params_log = [
                    np.array(layer.log_param.scale),
                    np.array(layer.log_param.shift),
                ]
                params_log_numpy = params_log

                shape = self.caffe_net.blobs[layer.bottom[0]].data.shape
                mul_layer.generate_params(params_log_numpy, shape)
                mul_layer.generate_node()

                self._node_post_process(mul_layer)

                add_layer = ops.AddLayer(layer, "_add")
                add_out_name = layer.name + "_add_out"
                add_layer._in_names.append(mul_out_name)
                add_layer._out_names.append(add_out_name)

                add_layer.generate_params(params_log_numpy, shape)
                add_layer.generate_node()

                self._node_post_process(add_layer)

                log_layer = ops.LogLayer(layer, "_log")
                log_layer._in_names.append(add_out_name)
                log_layer._out_names.extend(list(layer.top))

                log_layer.generate_node()

                self._node_post_process(log_layer)
            elif layer.type == "Power":
                # power layer = (mul + add) ^ power
                mul_layer = ops.MulLayer(layer, "_mul")
                mul_out_name = layer.name + "_mul_out"

                mul_layer._in_names.extend(list(layer.bottom))
                mul_layer._out_names.append(mul_out_name)

                params_power = [
                    np.array(layer.power_param.scale),
                    np.array(layer.power_param.shift),
                ]
                params_power_numpy = params_power

                shape = self.caffe_net.blobs[layer.bottom[0]].data.shape
                mul_layer.generate_params(params_power_numpy, shape)
                mul_layer.generate_node()

                self._node_post_process(mul_layer)

                add_layer = ops.AddLayer(layer, "_add")
                add_out_name = layer.name + "_add_out"
                add_layer._in_names.append(mul_out_name)
                add_layer._out_names.append(add_out_name)

                add_layer.generate_params(params_power_numpy, shape)
                add_layer.generate_node()

                self._node_post_process(add_layer)

                pow_layer = ops.PowerLayer(layer, "_power")
                pow_layer._in_names.append(add_out_name)
                pow_layer._out_names.extend(list(layer.top))
                params_power = np.array(layer.power_param.power)
                pow_layer.generate_params(params_power)
                pow_layer.generate_node()

                self._node_post_process(pow_layer)
            elif layer.type == "BNLL":
                bnll_layer = ops.BnllLayer(layer)
                bnll_layer._in_names.extend(list(layer.bottom))
                bnll_layer._out_names.extend(list(layer.top))
                bnll_layer.generate_node()

                self._node_post_process(bnll_layer)
            elif layer.type == "SELU":
                selu_layer = ops.SeluLayer(layer)
                selu_layer._in_names.extend(list(layer.bottom))
                selu_layer._out_names.extend(list(layer.top))
                selu_layer.generate_node()

                self._node_post_process(selu_layer)
            elif layer.type == "Sqrt":
                sqrt_layer = ops.SqrtLayer(layer)
                sqrt_layer._in_names.extend(list(layer.bottom))
                sqrt_layer._out_names.extend(list(layer.top))
                sqrt_layer.generate_node()

                self._node_post_process(sqrt_layer)
            elif layer.type == "Cos":
                cosine_layer = ops.CosineLayer(layer)

                cosine_layer._in_names.extend(list(layer.bottom))
                cosine_layer._out_names.extend(list(layer.top))
                cosine_layer.generate_node()

                self._node_post_process(cosine_layer)
            elif layer.type == "CReLU":
                relu_layer = ops.ReluLayer(layer, "_relu")
                relu_output_name = layer.name + "_relu_out"
                relu_layer._in_names.extend(list(layer.bottom))
                relu_layer._out_names.append(relu_output_name)
                relu_layer.generate_node()

                self._node_post_process(relu_layer)

                neg_layer = ops.NegLayer(layer, "_neg")
                neg_output_name = layer.name + "_neg_out"
                neg_layer._in_names.extend(list(layer.bottom))
                neg_layer._out_names.append(neg_output_name)
                neg_layer.generate_node()

                self._node_post_process(neg_layer)

                relu_neg_layer = ops.ReluLayer(layer, "_relu_neg")
                relu_neg_output_name = layer.name + "_relu_neg_out"
                relu_neg_layer._in_names.append(neg_output_name)
                relu_neg_layer._out_names.append(relu_neg_output_name)
                relu_neg_layer.generate_node()

                self._node_post_process(relu_neg_layer)

                concat_layer = ops.ConcatLayer(layer)
                relu_neg_output_name = layer.name + "_relu_neg_out"
                concat_layer._in_names.extend([relu_output_name, relu_neg_output_name])
                concat_layer._out_names.extend(list(layer.top))
                concat_layer.generate_node(layer.crelu_param.axis)

                self._node_post_process(concat_layer)
            elif layer.type == "AbsVal":
                abs_layer = ops.AbsLayer(layer)
                abs_layer._in_names.extend(list(layer.bottom))
                abs_layer._out_names.extend(list(layer.top))
                abs_layer.generate_node()

                self._node_post_process(abs_layer)
            elif layer.type == "Exp":
                # log layer = log (mul + add)
                mul_layer = ops.MulLayer(layer, "_mul")
                mul_out_name = layer.name + "_mul_out"

                mul_layer._in_names.extend(list(layer.bottom))
                mul_layer._out_names.append(mul_out_name)

                params_exp = [
                    np.array(layer.exp_param.scale),
                    np.array(layer.exp_param.shift),
                ]
                params_exp_numpy = params_exp

                shape = self.caffe_net.blobs[layer.bottom[0]].data.shape
                mul_layer.generate_params(params_exp_numpy, shape)
                mul_layer.generate_node()

                self._node_post_process(mul_layer)

                add_layer = ops.AddLayer(layer, "_add")
                add_layer._in_names.append(mul_out_name)

                add_out_name = layer.name + "_add_out"
                add_layer._out_names.append(add_out_name)

                add_layer.generate_params(params_exp_numpy, shape)
                add_layer.generate_node()

                self._node_post_process(add_layer)

                if layer.exp_param.base == -1:
                    exp_layer = ops.ExpLayer(layer, "_exp")
                    exp_layer._in_names.append(add_out_name)
                    exp_layer._out_names.extend(list(layer.top))

                    exp_layer.generate_node()

                    self._node_post_process(exp_layer)
                else:
                    power_layer = ops.PowerLayer(layer, "_power")
                    params_power = np.array(layer.exp_param.base)

                    power_layer.generate_params(params_power)
                    power_layer._in_names.append(add_out_name)
                    power_layer._out_names.extend(list(layer.top))

                    power_layer.generate_node()

                    self._node_post_process(power_layer)
            elif layer.type == "TanH":
                tanh_layer = ops.TanhLayer(layer)
                tanh_layer._in_names.extend(list(layer.bottom))
                tanh_layer._out_names.extend(list(layer.top))
                tanh_layer.generate_node()

                self._node_post_process(tanh_layer)
            elif layer.type == "Tile":
                tile_layer = ops.TileLayer(layer)
                tile_layer._in_names.extend(list(layer.bottom))
                tile_layer._out_names.extend(list(layer.top))

                shape = self.caffe_net.blobs[layer.bottom[0]].data.shape

                tile_layer.generate_node(shape)

                self._node_post_process(tile_layer)
            elif layer.type == "PReLU":
                prelu_layer = ops.PReluLayer(layer)
                prelu_layer._in_names.extend(list(layer.bottom))
                prelu_layer._out_names.extend(list(layer.top))
                params_prelu = self.caffe_net.params[layer.name]
                params_prelu_numpy = self._param_to_numpy(params_prelu)

                shape = self.caffe_net.blobs[layer.bottom[0]].data.shape

                prelu_layer.generate_node(params_prelu_numpy[0], shape)

                self._node_post_process(prelu_layer)
            elif layer.type == "Sin":
                sine_layer = ops.SineLayer(layer)
                sine_layer._in_names.extend(list(layer.bottom))
                sine_layer._out_names.extend(list(layer.top))
                sine_layer.generate_node()

                self._node_post_process(sine_layer)
            elif layer.type == "Slice":
                shape = self.caffe_net.blobs[layer.bottom[0]].data.shape
                slice_point = list(layer.slice_param.slice_point)
                axes = layer.slice_param.axis

                if len(slice_point) != 0:
                    assert len(list(layer.top)) == (len(slice_point) + 1)
                else:
                    assert shape[axes] % len(list(layer.top)) == 0
                    slice_point = (
                        np.linspace(0, shape[axes], len((layer.top)), endpoint=False)
                        .astype(int)[1:]
                        .tolist()
                    )

                start_index = [0]
                slice_point = start_index + slice_point + [shape[axes]]

                for idx in range(len(list(layer.top))):
                    slice_layer = ops.SliceLayer(layer, "_slice_" + str(idx))
                    slice_layer._in_names.extend(list(layer.bottom))
                    slice_layer._out_names.append(list(layer.top)[idx])
                    start_end = slice_point[idx : idx + 2]

                    params_slice = [
                        np.array([start_end[0]]),
                        np.array([start_end[1]]),
                        np.array([axes]),
                    ]

                    slice_layer.generate_params(params_slice)
                    slice_layer.generate_node()

                    self._node_post_process(slice_layer)
            elif layer.type == "ReLU6":
                relu_layer = ops.ReluLayer(layer)
                relu_layer_out_name = layer.name + "_relu_out"
                relu_layer._in_names.extend(list(layer.bottom))
                relu_layer._out_names.append(relu_layer_out_name)

                relu_layer.generate_node()

                self._node_post_process(relu_layer)

                clip_layer = ops.ClipLayer(layer, "_clip")
                clip_layer._in_names.append(relu_layer_out_name)
                clip_layer._out_names.extend(list(layer.top))
                params_clip = [np.array(np.NINF), np.array(layer.relu6_param.threshold)]
                clip_layer.generate_params(params_clip)
                clip_layer.generate_node()

                self._node_post_process(clip_layer)
            elif layer.type == "ELU":
                elu_layer = ops.EluLayer(layer)
                elu_layer._in_names.extend(list(layer.bottom))
                elu_layer._out_names.extend(list(layer.top))

                elu_layer.generate_node()

                self._node_post_process(elu_layer)
            elif layer.type == "Upsample":
                logging.warning(
                    "Upsample only supports nearest upsample with same scale"
                )
                scale = layer.upsample_param.scale
                params_upsample = [
                    np.array([]),
                    np.array([1.0, 1.0, scale, scale], dtype=np.float32),
                ]
                upsample_layer = ops.UpsampleLayer(layer)
                upsample_layer._in_names.extend(list(layer.bottom))
                upsample_layer._out_names.extend(list(layer.top))

                upsample_layer.generate_params(params_upsample)
                upsample_layer.generate_node()

                self._node_post_process(upsample_layer)
            elif layer.type == "UpsampleBN":
                stride = layer.upsample_param.scale
                shape = self.caffe_net.blobs[layer.bottom[0]].data.shape
                w = np.zeros((stride, stride))
                w[0, 0] = 1
                w = np.expand_dims(w, axis=(0, 1))
                w = np.repeat(w, shape[1], axis=0)
                params_upsample = [w]
                upsamplebn_layer = ops.UpsampleBNLayer(layer)
                upsamplebn_layer._in_names.extend(list(layer.bottom))
                upsamplebn_layer._out_names.extend(list(layer.top))

                upsamplebn_layer.generate_params(params_upsample)
                upsamplebn_layer.generate_node(shape)

                self._node_post_process(upsamplebn_layer)
            elif layer.type == "Flatten":
                shape = list(self.caffe_net.blobs[layer.bottom[0]].data.shape)
                start_axis = layer.flatten_param.axis
                end_axis = layer.flatten_param.end_axis

                if start_axis < 0:
                    start_axis = len(shape) + start_axis
                if end_axis < 0:
                    end_axis = len(shape) + end_axis

                flatten_layer = ops.FlattenLayer(layer)
                flatten_layer._in_names.extend(list(layer.bottom))
                flatten_layer._out_names.extend(list(layer.top))
                shape_new = shape[:start_axis] + [-1] + shape[end_axis + 1 :]

                flatten_layer.generate_node(shape_new)

                self._node_post_process(flatten_layer)
            elif layer.type == "Reduction":
                shape = list(self.caffe_net.blobs[layer.bottom[0]].data.shape)
                if layer.reduction_param.operation == 2:
                    abs_layer = ops.AbsLayer(layer, "_abs")
                    abs_layer_out_name = layer.name + "_abs_out"
                    abs_layer._in_names.extend(list(layer.bottom))
                    abs_layer._out_names.append(abs_layer_out_name)

                    abs_layer.generate_node()

                    self._node_post_process(abs_layer)

                    if layer.reduction_param.coeff != 1.0:
                        reduction_layer = ops.ReductionLayer(layer, "_reduction")
                        reduction_layer_out_name = layer.name + "_reduction_out"
                        reduction_layer._in_names.append(abs_layer_out_name)
                        reduction_layer._out_names.append(reduction_layer_out_name)

                        reduction_layer.generate_node(shape)

                        self._node_post_process(reduction_layer)
                        mul_layer = ops.MulLayer(layer, "_mul")

                        params_scale_numpy = np.array([layer.reduction_param.coeff])

                        mul_out_name = layer.name + "_mul_out"

                        mul_layer._in_names.append(reduction_layer_out_name)
                        mul_layer._out_names.extend(list(layer.top))

                        mul_layer.generate_params(params_scale_numpy)
                        mul_layer.generate_node()

                        self._node_post_process(mul_layer)
                    else:
                        reduction_layer = ops.ReductionLayer(layer, "_reduction")
                        reduction_layer._in_names.append(abs_layer_out_name)
                        reduction_layer._out_names.extend(list(layer.top))

                        reduction_layer.generate_node(shape)

                        self._node_post_process(reduction_layer)
                else:
                    if layer.reduction_param.coeff != 1.0:
                        reduction_layer = ops.ReductionLayer(layer, "_reduction")
                        reduction_layer_out_name = layer.name + "_reduction_out"
                        reduction_layer._in_names.extend(list(layer.bottom))
                        reduction_layer._out_names.append(reduction_layer_out_name)

                        reduction_layer.generate_node(shape)

                        self._node_post_process(reduction_layer)
                        mul_layer = ops.MulLayer(layer, "_mul")

                        params_scale_numpy = np.array([layer.reduction_param.coeff])

                        mul_out_name = layer.name + "_mul_out"

                        mul_layer._in_names.append(reduction_layer_out_name)
                        mul_layer._out_names.extend(list(layer.top))

                        mul_layer.generate_params(params_scale_numpy)
                        mul_layer.generate_node()

                        self._node_post_process(mul_layer)
                    else:
                        reduction_layer = ops.ReductionLayer(layer)
                        reduction_layer._in_names.extend(list(layer.bottom))
                        reduction_layer._out_names.extend(list(layer.top))

                        reduction_layer.generate_node(shape)

                        self._node_post_process(reduction_layer)
            else:
                raise Exception("unsupported layer type: {}".format(layer.type))

        for input_name in self.caffe_net.inputs:
            shape = self.caffe_net.blobs[input_name].shape
            shape_str = " ".join(str(e) for e in shape)
            logging.info("caffe input: " + input_name + " shape: " + shape_str)

            input_layer = ops.InputLayer()
            input_layer._generate_input(input_name, shape)
            self.in_tensor_value_info.extend(input_layer._in_tensor_value_info)

        for output_name in self.caffe_net.outputs:
            shape = self.caffe_net.blobs[output_name].shape
            shape_str = " ".join(str(e) for e in shape)
            if output_name in self.inplace_dict.keys():
                last_key = list(self.inplace_dict[output_name].keys())[-1]
                output_name = self.inplace_dict[output_name][last_key]["new_output"]

            output_layer = ops.OutputLayer()
            output_layer._generate_output(output_name, shape)
            logging.info("caffe output: " + output_name + " shape: " + shape_str)
            self.out_tensor_value_info.extend(output_layer._out_tensor_value_info)

        graph_def = helper.make_graph(
            self.nodes,
            self.onnx_file_name,
            self.in_tensor_value_info,
            self.out_tensor_value_info,
            self.init_tensor,
        )

        self.model_def = helper.make_model(graph_def, producer_name="caffe")
        self._freeze()
        checker.check_model(self.model_def)
        logging.info("onnx model conversion completed")

    def save(self):
        logging.info(
            "onnx model saved to "
            + self.save_path
            + os.sep
            + self.onnx_file_name
            + ".onnx"
        )
        save(self.model_def, self.save_path + os.sep + self.onnx_file_name + ".onnx")

    def test(self):
        onnx_rt_dict = {}
        for input_name in self.caffe_net.inputs:
            shape = self.caffe_net.blobs[input_name].shape
            input_data = np.random.randn(*shape).astype(np.float32)
            shape_str = " ".join(str(e) for e in shape)
            self.caffe_net.blobs[input_name].data[...] = input_data
            logging.info("caffe input: " + input_name + "shape: " + shape_str)
            onnx_rt_dict[input_name] = input_data

        pred = self.caffe_net.forward()
        sess = rt.InferenceSession(self.model_def.SerializeToString())
        onnx_outname = [output.name for output in sess.get_outputs()]
        res = sess.run(onnx_outname, onnx_rt_dict)
        caffe_outname = self.caffe_net.outputs

        assert len(onnx_outname) == len(caffe_outname)
        for idx in range(len(onnx_outname)):
            np.testing.assert_allclose(
                pred[caffe_outname[idx]], res[idx], rtol=1e-02, atol=1e-03
            )

    def _node_post_process(self, onnx_layer):
        self.nodes.append(onnx_layer._node)
        self.in_tensor_value_info.extend(onnx_layer._in_tensor_value_info)
        self.init_tensor.extend(onnx_layer._init_tensor)

    def _print_inplace_dict(self):
        import json

        print(json.dumps(self.inplace_dict, indent=4))

    def _update_inplace_dict(self, layer, input_name, output_name):
        assert len(layer.top) == 1
        if layer.top[0] not in self.inplace_dict.keys():
            self.inplace_dict[layer.top[0]] = {}

        self.inplace_dict[layer.top[0]][layer.name] = {}
        self.inplace_dict[layer.top[0]][layer.name]["orig_input"] = layer.top[0]
        self.inplace_dict[layer.top[0]][layer.name]["orig_output"] = layer.top[0]
        self.inplace_dict[layer.top[0]][layer.name]["new_input"] = input_name
        self.inplace_dict[layer.top[0]][layer.name]["new_output"] = output_name

    def _load_caffe_prototxt(self):
        net = caffe_pb2.NetParameter()
        with open(self.proto_file) as f:
            text_format.Merge(f.read(), net)

        return net

    def _load_caffe_weight(self):
        weight = caffe_pb2.NetParameter()
        with open(self.weight_file, "rb") as f:
            weight.ParseFromString(f.read())

        return weight

    def _load_net(self):
        caffe_net = caffe.Net(self.proto_file, caffe.TEST, weights=self.weight_file)

        return caffe_net

    def _get_net_length(self):

        return len(self.caffe_net._layer_names)

    def _get_layer_index(self, layer):
        idx = list(self.caffe_net._layer_names).index(layer.name)

        return idx

    def _param_to_numpy(self, params):
        params_numpy = [p.data for p in params]

        return params_numpy

    def _freeze(self):
        logging.info("removing not constant initializers from model")
        inputs = self.model_def.graph.input
        name_to_input = {}
        for input in inputs:
            name_to_input[input.name] = input

        for initializer in self.model_def.graph.initializer:
            if initializer.name in name_to_input:
                inputs.remove(name_to_input[initializer.name])
