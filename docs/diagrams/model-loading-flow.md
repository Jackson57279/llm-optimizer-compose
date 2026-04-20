# Model Loading & Registry Flow

```mermaid
flowchart TD
    subgraph Entry["🚪 Entry Points"]
        FromPre["from_pretrained()<br/>• HuggingFace ID<br/>• Auto-download"]
        FromGGUF["from_gguf()<br/>• Local GGUF file<br/>• Direct load"]
        New["__init__()<br/>• Manual config<br/>• Lazy load"]
    end

    subgraph LoadProcess["📥 Loading Process (_load_model)"]
        direction TB
        GetInfo["Get model info<br/>from registry"]
        
        subgraph ArchCheck["🏗️ Architecture Detection"]
            MoECheck{Is MoE?}
            Layers["Detect n_layers<br/>from model ID"]
            ContextCalc["compute_moe_context()<br/>AirLLM-style sizing"]
        end
        
        subgraph Quant["🎯 Quantization Selection"]
            UserQuant{User specified?}
            Select["MemoryManager<br/>select_quantization()<br/>Based on available RAM"]
        end
        
        subgraph Cache["💾 Cache Resolution"]
            Cached{GGUF cached?}
            Download{Download from<br/>HuggingFace?}
            Convert["Auto-convert<br/>SafeTensors→GGUF"]
        end
    end

    subgraph Registry["📚 Model Registry"]
        RegistryObj["registry.py<br/>• Model metadata<br/>• Cache paths"]
        ModelInfo["ModelInfo<br/>• params_billions<br/>• max_context<br/>• is_moe"]
    end

    subgraph DownloadFlow["⬇️ Download Flow"]
        direction TB
        TryRepos["Try repo variants:<br/>1. unsloth/{model}<br/>2. {model}-GGUF"]
        Subdir["Try subdirectory<br/>{bit-depth}/{quant}"]
        Flat["Try flat structure<br/>root/*.gguf"]
        MultiShard["Multi-shard?<br/>Download all shards"]
    end

    subgraph Init["⚡ Initialization"]
        CreateEngine["Create InferenceEngine<br/>with backend config"]
        InitEngine["engine.initialize()"]
        RegMemory["Register memory usage<br/>• model weights<br/>• KV cache"]
    end

    FromPre --> GetInfo
    FromGGUF --> CreateEngine
    New --> |later| GetInfo
    
    GetInfo --> RegistryObj
    RegistryObj --> ModelInfo
    
    GetInfo --> MoECheck
    MoECheck -->|Yes| Layers
    Layers --> ContextCalc
    MoECheck -->|No| UserQuant
    ContextCalc --> UserQuant
    
    UserQuant -->|No| Select
    UserQuant -->|Yes| Cached
    Select --> Cached
    
    Cached -->|Yes| CreateEngine
    Cached -->|No| Download
    
    Download -->|Success| TryRepos
    Download -->|Fail| Convert
    
    TryRepos --> Subdir
    Subdir -->|Not found| Flat
    Flat -->|Found| MultiShard
    
    Convert --> CreateEngine
    MultiShard --> CreateEngine
    
    CreateEngine --> InitEngine
    InitEngine --> RegMemory
    RegMemory --> Ready(["Model Ready"])

    style Entry fill:#e3f2fd
    style LoadProcess fill:#fff3e0
    style Registry fill:#e8f5e9
    style DownloadFlow fill:#fce4ec
    style Init fill:#f3e5f5
    style Ready fill:#e8f5e9
```

## Architecture Mapping

| Model Series | n_layers | Context | Notes |
|--------------|----------|---------|-------|
| MiniMax M2.x | 62 | 192K | Default for M2.7 |
| MiniMax-01 | 80 | 4M | Text/VL-01, M1 |
| Llama 4 | 48-80 | Varies | Scout/Maverick |
| Qwen3 | 24-80 | 128K | Size-dependent |
| Kimi K2.5 | 64 | 256K | 1M context capable |

## Cache Directory Structure

```
{cache_dir}/
├── gguf/
│   ├── MiniMaxAI/MiniMax-M2.7/
│   │   └── Q4_K_M/
│   │       └── model.gguf
│   └── unsloth/
│       └── ...
├── huggingface/
│   └── ...
└── registry.json
```
