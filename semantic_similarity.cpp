#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <thread>
#include <mutex>

// Função para calcular a similaridade de cosseno entre dois vetores
float cosine_similarity(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    float dot_product = 0.0, mag1 = 0.0, mag2 = 0.0;
    for (size_t i = 0; i < vec1.size(); i++) {
        dot_product += vec1[i] * vec2[i];
        mag1 += vec1[i] * vec1[i];
        mag2 += vec2[i] * vec2[i];
    }
    if (mag1 == 0.0f || mag2 == 0.0f)
        return 0.0f;
    return dot_product / (std::sqrt(mag1) * std::sqrt(mag2));
}

// Função para executar o modelo ONNX com os dados fornecidos
std::vector<float> run_model(Ort::Session& session, const int64_t* input_ids, const int64_t* attention_mask, size_t size) {
    // Dimensões do tensor
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(size)};

    // Criar OrtMemoryInfo
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Criar tensores para input_ids e attention_mask
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(input_ids), size, input_shape.data(), input_shape.size());
    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(attention_mask), size, input_shape.data(), input_shape.size());

    // Executar o modelo ONNX
    const char* input_names[] = {"input_ids", "attention_mask"};
    const char* output_names[] = {"output"};
    Ort::Value input_tensors[] = {std::move(input_tensor), std::move(attention_mask_tensor)};
    
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, input_tensors, 2, output_names, 1);

    // Pegar a saída do tensor
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    size_t size_out = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
    
    return std::vector<float>(floatarr, floatarr + size_out);
}

// Função para calcular a similaridade entre duas frases e retornar a similaridade
float compute_similarity(Ort::Session& session, const int64_t* input_ids1, const int64_t* attention_mask1, size_t size1,
                                    const int64_t* input_ids2, const int64_t* attention_mask2, size_t size2) {

    // Obter os embeddings das frases
    std::vector<float> embedding1 = run_model(session, input_ids1, attention_mask1, size1);
    std::vector<float> embedding2 = run_model(session, input_ids2, attention_mask2, size2);

    // Calcular a similaridade do cosseno
    float similarity = cosine_similarity(embedding1, embedding2);
    return similarity * 100.0f; // Retornar como percentual
}

// Função para rodar o cálculo de similaridade em paralelo e retornar os resultados para o Python
extern "C" void run_parallel_similarity(const int64_t** input_ids1_list, const int64_t** attention_mask1_list, const size_t* size1_list,
                                        const int64_t** input_ids2_list, const int64_t** attention_mask2_list, const size_t* size2_list,
                                        size_t num_threads, const char* model_path, float* result_list) {

    // Inicializar o ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SemanticSimilarity");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path, session_options);

    // Criar múltiplas threads
    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads; ++i) {
        threads.push_back(std::thread([&, i]() {
            result_list[i] = compute_similarity(session,
                                                input_ids1_list[i], attention_mask1_list[i], size1_list[i],
                                                input_ids2_list[i], attention_mask2_list[i], size2_list[i]);
        }));
    }

    // Esperar que todas as threads terminem
    for (auto& thread : threads) {
        thread.join();
    }
}
