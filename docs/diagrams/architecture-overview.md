# Miniforge Architecture Overview

```mermaid
flowchart TB
    subgraph PublicAPI["📦 Public API (miniforge/__init__.py)"]
        Miniforge["Miniforge<br/>High-level interface"]
        InferenceEngine["InferenceEngine<br/>Unified backend interface"]
        M7Config["M7Config<br/>GMKtech M7 optimized config"]
    end

    subgraph Core["🔧 Core Engine (core/)"]
        Engine["InferenceEngine<br/>• Backend abstraction<br/>• Generation<br/>• Chat completion"]
        MemoryMgr["MemoryManager<br/>• RAM monitoring<br/>• KV cache sizing<br/>• MoE context calc"]
        
        subgraph Backends["Backends (core/backends/)"]
            LlamaCpp["LlamaCppBackend<br/>• GGUF inference<br/>• llama.cpp binding<br/>• mmap/mlock support"]
            Transformers["TransformersBackend<br/>• HF transformers<br/>• Native PyTorch<br/>• Fallback"]
        end
    end

    subgraph Models["🤖 Models (models/)"]
        MiniMaxImpl["minimax.py<br/>• Model loading<br/>• GGUF download<br/>• Auto-conversion"]
        Registry["registry.py<br/>• Model registry<br/>• Cache management<br/>• Metadata"]
        GGUFConvert["gguf_convert.py<br/>• SafeTensors→GGUF<br/>• Auto-quantization"]
    end

    subgraph Generation["✨ Generation (generation/)"]
        Tools["tools.py<br/>Tool calling & execution"]
        Streaming["streaming.py<br/>Streaming response handler"]
    end

    subgraph Multimodal["🖼️ Multimodal (multimodal/)"]
        Vision["vision.py<br/>VisionProcessor<br/>Image understanding"]
    end

    subgraph Utils["⚙️ Utils (utils/)"]
        Config["config.py<br/>M7Config dataclass<br/>• Hardware limits<br/>• Performance presets"]
        Monitoring["monitoring.py<br/>Performance metrics"]
        Download["download.py<br/>Model download utilities"]
    end

    subgraph External["🌐 External"]
        HFHub["HuggingFace Hub<br/>• Model download<br/>• GGUF repos"]
        LlamaCppLib["llama-cpp-python<br/>CPU inference engine"]
        TransformersLib["transformers + torch<br/>PyTorch backend"]
    end

    subgraph Hardware["💻 Target Hardware"]
        GMKtech["GMKtech M7<br/>• Ryzen 7 PRO 6850H<br/>• 28GB RAM<br/>• 4GB VRAM (iGPU)"]
    end

    %% Flow connections
    Miniforge --> MiniMaxImpl
    InferenceEngine --> Engine
    M7Config --> Config

    MiniMaxImpl --> Engine
    MiniMaxImpl --> MemoryMgr
    MiniMaxImpl --> Registry
    MiniMaxImpl --> Tools
    MiniMaxImpl --> Vision

    Engine --> LlamaCpp
    Engine --> Transformers
    
    LlamaCpp <---> LlamaCppLib
    Transformers <---> TransformersLib

    Registry --> HFHub
    GGUFConvert --> LlamaCppLib
    
    MiniMaxImpl --> GGUFConvert

    Config --> GMKtech
    MemoryMgr --> GMKtech

    style PublicAPI fill:#e1f5fe
    style Core fill:#fff3e0
    style Models fill:#f3e5f5
    style Generation fill:#e8f5e9
    style Multimodal fill:#fce4ec
    style Utils fill:#f5f5f5
    style External fill:#fff8e1
    style Hardware fill:#ffebee
```

## Architecture Summary

| Layer | Purpose | Key Components |
|-------|---------|----------------|
| **Public API** | User-facing interface | `Miniforge`, `InferenceEngine`, `M7Config` |
| **Core** | Inference & memory management | Backends, memory manager, engine |
| **Models** | Model loading & conversion | Registry, GGUF conversion, MiniMax model |
| **Generation** | Output generation features | Tool calling, streaming |
| **Multimodal** | Vision capabilities | Vision processor |
| **Utils** | Configuration & helpers | M7Config, monitoring, downloads |
