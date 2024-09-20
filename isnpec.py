import onnxruntime as ort

# Carregar o modelo ONNX
onnx_model_path = "onnx/minilm.onnx"
session = ort.InferenceSession(onnx_model_path)

# Listar os nomes das saídas
for output in session.get_outputs():
    print(f"Nome da saída: {output.name}")

