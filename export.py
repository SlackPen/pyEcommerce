import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# Carregar o modelo pré-treinado
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Colocar o modelo em modo de avaliação
model.eval()

# Caminho de saída para o arquivo ONNX
output_dir = Path("onnx")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "minilm.onnx"

# Texto de exemplo para exportação
dummy_input = tokenizer("Texto de exemplo para exportar o modelo ONNX.", return_tensors="pt")

# Exportar o modelo para ONNX
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),  # Entradas para o modelo
    output_path,
    export_params=True,
    opset_version=11,
    input_names=["input_ids", "attention_mask"],
    output_names=["output"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print(f"Modelo ONNX salvo em: {output_path}")

