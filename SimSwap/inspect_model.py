import inspect
from insightface.model_zoo import model_zoo
import os

def inspect_model():
    onnx_file = './insightface_func/models/antelope/det_10g.onnx'
    model = model_zoo.get_model(onnx_file)
    print(f"Model type: {type(model)}")
    print(f"Detect method signature: {inspect.signature(model.detect)}")
    print(f"Prepare method signature: {inspect.signature(model.prepare)}")

if __name__ == "__main__":
    inspect_model()
