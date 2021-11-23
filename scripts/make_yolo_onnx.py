import torch
from libs.models import Darknet



def convert_to_onnx(cfgfile: str, weightsfile: str, IMAGE_WIDTH: int, IMAGE_HEIGHT: int):
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)

    x = torch.randn((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), requires_grad=True)

    onnx_filename = '../models/yolo.onnx'

    print('Export the onnx model ...')

    torch.onnx.export(model,
                      x,
                      onnx_filename,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['boxes', 'confs'],
                      dynamic_axes={
                          'input': {0: 'batch_size'},
                          'boxes': {0: 'batch_size'},
                          'confs': {0: 'batch_size'},
                      })

    print('Onnx model exporting done!')


if __name__ == '__main__':
    convert_to_onnx("./yolov4-mish.cfg", "./yolov4-mish.weights", 512, 512)
