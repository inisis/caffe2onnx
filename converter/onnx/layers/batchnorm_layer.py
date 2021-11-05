import logging
from onnx import helper
from onnx import TensorProto as tp


from layers.base_layer import BaseLayer


class BatchNormLayer(BaseLayer):
    def __init__(self, layer, name=None):
        super(BatchNormLayer, self).__init__(layer, name)

    def get_batchnorm_attr(self):
        attr_dict = {"epsilon": 1e-5, "momentum": 0.999}

        attr_dict["epsilon"] = self._layer.batch_norm_param.eps
        attr_dict["momentum"] = self._layer.batch_norm_param.moving_average_fraction

        return attr_dict

    def create_batchnorm_scale(self, params):
        param_name = self._layer.name + "_scale"

        param_type = tp.FLOAT
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

    def create_batchnorm_bias(self, params):
        param_name = self._layer.name + "_b"

        param_type = tp.FLOAT
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

    def create_batchnorm_scale_param(self, params):
        self.create_batchnorm_scale(params[0])
        self.create_batchnorm_bias(params[1])

    def create_batchnorm_mean(self, params):
        param_name = self._layer.name + "_mean"

        param_type = tp.FLOAT
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

    def create_batchnorm_var(self, params):
        param_name = self._layer.name + "_var"

        param_type = tp.FLOAT
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

    def create_batchnorm_param(self, params):
        if len(params) == 3:
            self.create_batchnorm_mean(params[0] / params[2])
            self.create_batchnorm_var(params[1] / params[2])
        else:
            raise Exception(
                "unsupported batchnorm param length: {}".format(len(params))
            )

    def generate_node(self):
        attr_dict = self.get_batchnorm_attr()
        logging.debug(attr_dict)

        node = helper.make_node(
            "BatchNormalization",
            self._in_names,
            self._out_names,
            self._layer.name,
            **attr_dict
        )
        logging.info("batchnorm_layer: " + self._layer.name + " created")

        self._node = node

    def generate_params(self, params_batchnorm, params_scale):
        self.create_batchnorm_scale_param(params_scale)
        self.create_batchnorm_param(params_batchnorm)
