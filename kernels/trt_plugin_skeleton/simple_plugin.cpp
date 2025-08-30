#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <vector>
#include <cstring>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>

using namespace nvinfer1;

// Forward declaration for CUTLASS integration
extern "C" void cutlass_gemm_stub(const float* A, const float* B, float* C, 
                                  int M, int N, int K, cudaStream_t stream);

/**
 * Simple TensorRT plugin that demonstrates how to call cuBLAS GEMM in enqueue.
 * This plugin supports both FP32 and FP16 precision modes.
 * 
 * Features:
 * - Dynamic shape support
 * - Multiple precision modes (FP32, FP16)
 * - CUTLASS integration (optional)
 * - Proper error handling and resource management
 */
class SimpleGemmPlugin : public IPluginV2DynamicExt {
public:
    SimpleGemmPlugin(int m = 0, int n = 0, int k = 0, DataType dtype = DataType::kFLOAT)
        : m_(m), n_(n), k_(k), dtype_(dtype), cublasHandle_(nullptr), use_cutlass_(false) {
        // Initialize cuBLAS handle
        cublasCreate(&cublasHandle_);
    }
    
    SimpleGemmPlugin(const void* data, size_t length) : cublasHandle_(nullptr), use_cutlass_(false) {
        // Deserialize parameters
        if (length >= sizeof(int) * 4) {
            const int* p = reinterpret_cast<const int*>(data);
            m_ = p[0]; 
            n_ = p[1]; 
            k_ = p[2];
            dtype_ = static_cast<DataType>(p[3]);
        }
        // Initialize cuBLAS handle
        cublasCreate(&cublasHandle_);
    }
    
    ~SimpleGemmPlugin() {
        if (cublasHandle_) {
            cublasDestroy(cublasHandle_);
        }
    }

    // IPluginV2 methods
    const char* getPluginType() const noexcept override { 
        return "SimpleGemmPlugin"; 
    }
    
    const char* getPluginVersion() const noexcept override { 
        return "1.0"; 
    }
    
    int getNbOutputs() const noexcept override { 
        return 1; 
    }

    // IPluginV2DynamicExt methods
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, 
                                 int nbInputs, IExprBuilder& exprBuilder) noexcept override {
        // Assume inputs[0] is A: [B, M, K], inputs[1] is B: [B, K, N] => output [B, M, N]
        DimsExprs out;
        out.nbDims = 3;
        out.d[0] = inputs[0].d[0]; // batch
        out.d[1] = inputs[0].d[1]; // M
        out.d[2] = inputs[1].d[2]; // N
        return out;
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, 
                                  int nbInputs, int nbOutputs) noexcept override {
        // Support FP32 and FP16 formats
        if (pos < nbInputs + nbOutputs) {
            return (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF) &&
                   inOut[pos].format == TensorFormat::kLINEAR;
        }
        return false;
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, 
                        const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override {
        // Store input/output descriptions for use in enqueue
        input_descs_.clear();
        output_descs_.clear();
        
        for (int i = 0; i < nbInputs; ++i) {
            input_descs_.push_back(in[i]);
        }
        for (int i = 0; i < nbOutputs; ++i) {
            output_descs_.push_back(out[i]);
        }
    }

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, 
                           const PluginTensorDesc* outputs, int nbOutputs) const noexcept override {
        return 0; // No workspace needed for this plugin
    }

    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs, 
                void* workspace, cudaStream_t stream) noexcept override {
        
        try {
            // Extract dimensions
            int B = inputDesc[0].dims.d[0];
            int M = inputDesc[0].dims.d[1];
            int K = inputDesc[0].dims.d[2];
            int N = inputDesc[1].dims.d[2];
            
            // Set cuBLAS stream
            cublasSetStream(cublasHandle_, stream);
            
            // Get data pointers
            const void* A = inputs[0];
            const void* Bptr = inputs[1];
            void* C = outputs[0];
            
            // Determine data type
            DataType dtype = inputDesc[0].type;
            
            // Perform GEMM based on data type
            cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;
            
            if (dtype == DataType::kFLOAT) {
                // FP32 GEMM
                const float* A_fp32 = reinterpret_cast<const float*>(A);
                const float* B_fp32 = reinterpret_cast<const float*>(Bptr);
                float* C_fp32 = reinterpret_cast<float*>(C);
                
                if (use_cutlass_) {
                    // Use CUTLASS if available
                    cutlass_gemm_stub(A_fp32, B_fp32, C_fp32, M, N, K, stream);
                } else {
                    // Use cuBLAS
                    const float alpha = 1.0f, beta = 0.0f;
                    long long strideA = static_cast<long long>(M) * K;
                    long long strideB = static_cast<long long>(K) * N;
                    long long strideC = static_cast<long long>(M) * N;
                    
                    stat = cublasSgemmStridedBatched(cublasHandle_,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K,
                        &alpha,
                        B_fp32, N, strideB,
                        A_fp32, K, strideA,
                        &beta,
                        C_fp32, N, strideC,
                        B);
                }
            } else if (dtype == DataType::kHALF) {
                // FP16 GEMM
                const half* A_fp16 = reinterpret_cast<const half*>(A);
                const half* B_fp16 = reinterpret_cast<const half*>(Bptr);
                half* C_fp16 = reinterpret_cast<half*>(C);
                
                const half alpha = __float2half(1.0f), beta = __float2half(0.0f);
                long long strideA = static_cast<long long>(M) * K;
                long long strideB = static_cast<long long>(K) * N;
                long long strideC = static_cast<long long>(M) * N;
                
                stat = cublasHgemmStridedBatched(cublasHandle_,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    B_fp16, N, strideB,
                    A_fp16, K, strideA,
                    &beta,
                    C_fp16, N, strideC,
                    B);
            }
            
            if (stat != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cuBLAS GEMM failed: " << stat << std::endl;
                return -1;
            }
            
            return 0;
            
        } catch (const std::exception& e) {
            std::cerr << "Plugin enqueue error: " << e.what() << std::endl;
            return -1;
        }
    }

    // Serialization
    size_t getSerializationSize() const noexcept override { 
        return sizeof(int) * 4; // m, n, k, dtype
    }
    
    void serialize(void* buffer) const noexcept override {
        int* p = reinterpret_cast<int*>(buffer);
        p[0] = m_; 
        p[1] = n_; 
        p[2] = k_;
        p[3] = static_cast<int>(dtype_);
    }

    // IPluginV2 methods
    void destroy() noexcept override { 
        delete this; 
    }
    
    IPluginV2DynamicExt* clone() const noexcept override { 
        return new SimpleGemmPlugin(m_, n_, k_, dtype_); 
    }
    
    void setPluginNamespace(const char* libNamespace) noexcept override { 
        namespace_ = libNamespace; 
    }
    
    const char* getPluginNamespace() const noexcept override { 
        return namespace_.c_str(); 
    }
    
    DataType getOutputDataType(int index, const DataType* inputTypes, 
                              int nbInputs) const noexcept override { 
        return inputTypes[0]; 
    }
    
    void attachToContext(cudnnContext* /*cudnn*/, cublasContext* cublas) noexcept override {
        if (cublas && !cublasHandle_) {
            cublasCreate(&cublasHandle_);
        }
    }
    
    void detachFromContext() noexcept override {
        if (cublasHandle_) {
            cublasDestroy(cublasHandle_);
            cublasHandle_ = nullptr;
        }
    }
    
    // Plugin-specific methods
    void setUseCUTLASS(bool use_cutlass) { use_cutlass_ = use_cutlass; }
    bool getUseCUTLASS() const { return use_cutlass_; }

private:
    int m_, n_, k_;
    DataType dtype_;
    cublasHandle_t cublasHandle_;
    std::string namespace_;
    bool use_cutlass_;
    std::vector<DynamicPluginTensorDesc> input_descs_;
    std::vector<DynamicPluginTensorDesc> output_descs_;
};

// Plugin Creator
class SimpleGemmPluginCreator : public IPluginCreator {
public:
    SimpleGemmPluginCreator() {}
    
    const char* getPluginName() const noexcept override {
        return "SimpleGemmPlugin";
    }
    
    const char* getPluginVersion() const noexcept override {
        return "1.0";
    }
    
    const PluginFieldCollection* getFieldNames() noexcept override {
        return &field_collection_;
    }
    
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override {
        try {
            // Parse plugin fields
            int m = 0, n = 0, k = 0;
            DataType dtype = DataType::kFLOAT;
            
            if (fc) {
                for (int i = 0; i < fc->nbFields; ++i) {
                    const PluginField& field = fc->fields[i];
                    if (strcmp(field.name, "m") == 0) {
                        m = *static_cast<const int*>(field.data);
                    } else if (strcmp(field.name, "n") == 0) {
                        n = *static_cast<const int*>(field.data);
                    } else if (strcmp(field.name, "k") == 0) {
                        k = *static_cast<const int*>(field.data);
                    } else if (strcmp(field.name, "dtype") == 0) {
                        dtype = static_cast<DataType>(*static_cast<const int*>(field.data));
                    }
                }
            }
            
            return new SimpleGemmPlugin(m, n, k, dtype);
        } catch (const std::exception& e) {
            std::cerr << "Failed to create plugin: " << e.what() << std::endl;
            return nullptr;
        }
    }
    
    IPluginV2* deserializePlugin(const char* name, const void* serialData, 
                                size_t serialLength) noexcept override {
        try {
            return new SimpleGemmPlugin(serialData, serialLength);
        } catch (const std::exception& e) {
            std::cerr << "Failed to deserialize plugin: " << e.what() << std::endl;
            return nullptr;
        }
    }
    
    void setPluginNamespace(const char* libNamespace) noexcept override {
        namespace_ = libNamespace;
    }
    
    const char* getPluginNamespace() const noexcept override {
        return namespace_.c_str();
    }

private:
    std::string namespace_;
    PluginFieldCollection field_collection_;
    std::vector<PluginField> fields_;
};

// Register plugin creator
REGISTER_TENSORRT_PLUGIN(SimpleGemmPluginCreator);

// CUTLASS integration guidance:
// - Add kernels/cutlass_prototype/ with a CUTLASS-based implementation and headers.
// - Provide CMake option -DUSE_CUTLASS=ON to include CUTLASS headers and call CUTLASS templates
//   instead of cuBLAS in enqueue().
// - The cutlass_gemm_stub function should be implemented in a separate file with actual CUTLASS calls.

