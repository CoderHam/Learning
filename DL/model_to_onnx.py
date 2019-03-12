import torch
import torch.onnx
import torch.nn as nn
import torchvision.models as models
import onnx

def torch_to_onnx(model_path=None, weights_path=None, model_state_dict=None, dummy_input=None, onnx_model_name="onnx_model_name"):
    # Load the model and weights from a file (.pt or .pth)
    model = torch.load(model_path)
    # # export to onnx
    torch.onnx.export(model, dummy_input, onnx_model_name+".onnx", verbose=False)
    del model

def test_torch(caffe_backend_test=False):
    model = models.alexnet(pretrained=True)
    # Create the right input shape (e.g. for an image)
    dummy_input = torch.randn(10, 3, 224, 224)
    # save model and weights in .pth file
    torch.save(model, "alexnet.pth")
    del model
    # export from .pth to .onnx
    torch_to_onnx(model_path="alexnet.pth", dummy_input=dummy_input, onnx_model_name="alexnet")
    del dummy_input
    # test loaded model
    onnx_model = onnx.load("alexnet.onnx")
    # Check that the IR is well formed
    onnx.checker.check_model(onnx_model)
    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(onnx_model.graph))

    if caffe_backend_test:
        import caffe2.python.onnx.backend as backend
        import numpy as np

        model_backend = backend.prepare(onnx_model, device="CUDA:0")
        import time
        s = time.time()
        inputs = np.random.randn(10, 3, 224, 224).astype(np.float32)
        outputs = model_backend.run(inputs)
        print("Time taken: ", time.time()-s)
        print(outputs[0])


if __name__ == "__main__":
    test_torch(caffe_backend_test=True)
