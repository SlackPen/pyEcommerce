import pandas as pd
from transformers import AutoTokenizer
import ctypes
import numpy as np
import time
import warnings
from tqdm import tqdm  # Biblioteca para barra de progresso

# Suprimir os avisos específicos do Hugging Face e do Pandas
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Caminho absoluto para o modelo ONNX
modelo = b"onnx/minilm.onnx"  # Caminho para o modelo ONNX

# Carregar o tokenizer da Hugging Face
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Função para tokenizar as frases e gerar input_ids e attention_mask
def prepare_inputs(sentence, max_length=14):
    inputs = tokenizer(sentence, return_tensors="np", padding="max_length", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].flatten()
    attention_mask = inputs["attention_mask"].flatten()
    return input_ids, attention_mask

# Carregar a biblioteca compartilhada (.so no Linux)
lib = ctypes.CDLL('./semantic_similarity.so')

# Definir o tipo de argumento e retorno da função C++
lib.run_parallel_similarity.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)), 
                                        ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)), 
                                        ctypes.POINTER(ctypes.c_size_t),
                                        ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)), 
                                        ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)), 
                                        ctypes.POINTER(ctypes.c_size_t),
                                        ctypes.c_size_t, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float)]
lib.run_parallel_similarity.restype = None

# Função para distribuir e calcular similaridade entre DataFrames A e B com cronômetro e barra de progresso
def calculate_similarity_parallel(df_a, df_b, max_length=14):
    num_a = len(df_a)
    num_b = len(df_b)

    # Inicializar o cronômetro
    start_time_total = time.time()

    # Tempo para preparação dos dados (tokenização)
    start_time_python = time.time()

    # Preparar todas as colunas do DataFrame A de uma vez (evitar fragmentação)
    colunas_similarity = [f"similarity_with_B_{row_b['codigo']}" for idx_b, row_b in df_b.iterrows()]
    df_a = pd.concat([df_a, pd.DataFrame(0.0, index=df_a.index, columns=colunas_similarity)], axis=1)

    # Marcar o fim da preparação das colunas
    end_time_python = time.time()
    time_python = end_time_python - start_time_python
    print(f"Tempo gasto no Python (preparação de colunas): {time_python:.4f} segundos")

    # Usar tqdm para barra de progresso
    with tqdm(total=num_a, desc="Processando DataFrame A", unit="linha") as pbar:
        for idx_a, row_a in df_a.iterrows():
            frases1 = [row_a['mensagem']] * num_b
            frases2 = list(df_b['texto'])

            # Preparar as listas de input_ids e attention_mask
            input_ids1_list = []
            attention_mask1_list = []
            input_ids2_list = []
            attention_mask2_list = []
            size1_list = []
            size2_list = []

            for i in range(num_b):
                # Preparar as entradas para cada par de frases
                input_ids1, attention_mask1 = prepare_inputs(frases1[i], max_length=max_length)
                input_ids2, attention_mask2 = prepare_inputs(frases2[i], max_length=max_length)

                input_ids1_list.append(input_ids1.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
                attention_mask1_list.append(attention_mask1.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
                input_ids2_list.append(input_ids2.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
                attention_mask2_list.append(attention_mask2.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))

                size1_list.append(len(input_ids1))
                size2_list.append(len(input_ids2))

            # Converter para ctypes
            input_ids1_list_ctypes = (ctypes.POINTER(ctypes.c_int64) * num_b)(*input_ids1_list)
            attention_mask1_list_ctypes = (ctypes.POINTER(ctypes.c_int64) * num_b)(*attention_mask1_list)
            input_ids2_list_ctypes = (ctypes.POINTER(ctypes.c_int64) * num_b)(*input_ids2_list)
            attention_mask2_list_ctypes = (ctypes.POINTER(ctypes.c_int64) * num_b)(*attention_mask2_list)
            size1_list_ctypes = (ctypes.c_size_t * num_b)(*size1_list)
            size2_list_ctypes = (ctypes.c_size_t * num_b)(*size2_list)

            # Array para armazenar os resultados de similaridade
            result_list = (ctypes.c_float * num_b)()

            # Medir o tempo das operações em C++ (cálculo de similaridade)
            start_time_cpp = time.time()

            # Chamar a função C++ para calcular similaridades em paralelo
            lib.run_parallel_similarity(input_ids1_list_ctypes, attention_mask1_list_ctypes, size1_list_ctypes,
                                        input_ids2_list_ctypes, attention_mask2_list_ctypes, size2_list_ctypes,
                                        num_b, modelo, result_list)

            # Marcar o fim das operações em C++
            end_time_cpp = time.time()
            time_cpp = end_time_cpp - start_time_cpp
            print(f"Tempo gasto no C++ (cálculo de similaridade para linha {idx_a}): {time_cpp:.4f} segundos")

            # Preencher os valores de similaridade no DataFrame A para cada linha de B
            for i, row_b in df_b.iterrows():
                df_a.at[idx_a, f'similarity_with_B_{row_b["codigo"]}'] = result_list[i]

            # Atualizar a barra de progresso
            pbar.update(1)

    # Tempo total de processamento
    end_time_total = time.time()
    time_total = end_time_total - start_time_total
    print(f"Tempo total de processamento: {time_total:.4f} segundos")

    return df_a

# Exemplo de uso com DataFrames A e B
if __name__ == "__main__":
    # DataFrame A: código, mensagem (500 linhas)
    df_a = pd.DataFrame({
        'codigo': range(500),
        'mensagem': ['mensagem exemplo A'] * 500
    })

    # DataFrame B: código, data, texto (1030 linhas)
    df_b = pd.DataFrame({
        'codigo': range(1030),
        'data': pd.date_range('2023-01-01', periods=1030),
        'texto': ['mensagem exemplo B'] * 1030
    })

    # Calcular similaridade entre DataFrames A e B
    df_a = calculate_similarity_parallel(df_a, df_b)

    # Exibir o DataFrame A com as colunas de similaridade
    print(df_a)
