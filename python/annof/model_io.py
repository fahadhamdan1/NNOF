import onnx
import numpy as np
import annof_core

def import_onnx(model_path):
    model = onnx.load(model_path)
    graph = model.graph
    tensors = []
    
    for init in graph.initializer:
        np_array = onnx.numpy_helper.to_array(init)
        tensor = annof_core.Tensor(list(np_array.shape))
        np.copyto(np.array(tensor.data(), dtype=np.float32), np_array.flatten())
        tensors.append(tensor)
    
    return tensors

def export_optimized_model(tensors, output_path):
    # still need to convert the tensors back to an ONNX model :(
    print(f"Exporting {len(tensors)} optimized tensors to {output_path}")