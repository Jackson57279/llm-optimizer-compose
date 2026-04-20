# Backend & Inference Flow

```mermaid
flowchart LR
    subgraph User["👤 User"]
        Request["Request<br/>• prompt/message<br/>• generation params<br/>• stream?"]
    end

    subgraph API["🔌 Public API"]
        Miniforge["Miniforge.chat()<br/>Miniforge.generate()"]
    end

    subgraph Engine["⚙️ InferenceEngine"]
        direction TB
        Init["initialize()<br/>Lazy backend init"]
        Generate["generate()<br/>Route to backend"]
        Chat["chat()<br/>Format messages"]
        Stream["_generate_stream()<br/>Async iterator"]
        Sync["_generate_sync()<br/>Await result"]
    end

    subgraph Backends["🔧 Backend Abstraction"]
        direction TB
        BackendInit["Backend.initialize()<br/>Load model"]
        
        subgraph LlamaCpp["LlamaCppBackend"]
            LlamaGen["generate()<br/>llama.create_completion()"]
            LlamaStream["generate_stream()<br/>Token iterator"]
            LlamaInfo["get_info()<br/>Model metadata"]
        end
        
        subgraph Transformers["TransformersBackend"]
            TransGen["generate()<br/>model.generate()"]
            TransStream["generate_stream()<br/>Streamer callback"]
            TransInfo["get_info()<br/>Config extract"]
        end
    end

    subgraph External["🌐 External Libraries"]
        LlamaLib["llama-cpp-python"]
        TransLib["transformers<br/>torch"]
    end

    subgraph Output["📤 Output"]
        Text["Text response"]
        StreamOut["Streaming tokens"]
    end

    Request --> Miniforge
    Miniforge --> Engine
    
    Engine --> Init
    Init -->|llama_cpp| LlamaCpp
    Init -->|transformers| Transformers
    
    Generate --> Sync
    Generate --> Stream
    
    Sync --> LlamaGen
    Sync --> TransGen
    Stream --> LlamaStream
    Stream --> TransStream
    
    LlamaGen --> LlamaLib
    LlamaStream --> LlamaLib
    TransGen --> TransLib
    TransStream --> TransLib
    
    LlamaLib --> Text
    LlamaStream --> StreamOut
    TransLib --> Text
    TransStream --> StreamOut

    style User fill:#e3f2fd
    style API fill:#e8f5e9
    style Engine fill:#fff3e0
    style LlamaCpp fill:#fce4ec
    style Transformers fill:#f3e5f5
    style External fill:#fff8e1
    style Output fill:#e1f5fe
```

## Backend Selection Logic

```mermaid
flowchart TD
    Start(["Initialize Engine"]) --> CheckBackend{Backend?}
    
    CheckBackend -->|llama_cpp| CheckGGUF{GGUF file<br/>available?}
    CheckBackend -->|transformers| TransInit["TransformersBackend<br/>• Load HF model<br/>• Use PyTorch"]
    
    CheckGGUF -->|Yes| LlamaInit["LlamaCppBackend<br/>• Load GGUF<br/>• mmap/mlock"]
    CheckGGUF -->|No| Convert{Auto-convert<br/>from HF?}
    
    Convert -->|Yes| ConvertProcess["gguf_convert.py<br/>SafeTensors → GGUF"]
    Convert -->|No| Fallback["Fallback to<br/>transformers"]
    ConvertProcess --> LlamaInit
    
    LlamaInit --> Ready(["Backend Ready"])
    TransInit --> Ready
    Fallback --> TransInit

    style Start fill:#e8f5e9
    style Ready fill:#e8f5e9
    style LlamaInit fill:#fce4ec
    style TransInit fill:#f3e5f5
```

## Key Design Decisions

1. **Lazy Initialization**: Backends loaded only when needed via `initialize()`
2. **Async-First**: All I/O operations are async to prevent blocking
3. **Backend Fallback**: llama.cpp preferred, transformers as fallback
4. **Streaming Support**: Unified streaming interface across backends
