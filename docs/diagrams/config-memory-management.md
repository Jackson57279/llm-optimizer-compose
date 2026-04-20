# Configuration & Memory Management

```mermaid
flowchart TB
    subgraph ConfigSrc["📋 Configuration Sources"]
        direction TB
        Default["M7Config defaults<br/>• 24GB max RAM<br/>• 8 threads<br/>• Q4_K_M quant"]
        YAML["YAML file<br/>• ~/.config/miniforge/config.yaml<br/>• ./miniforge.yaml"]
        Code["Code override<br/>M7Config(param=value)"]
        Preset["Performance preset<br/>• speed<br/>• balanced<br/>• memory<br/>• quality<br/>• moe"]
    end

    subgraph M7Config["⚙️ M7Config Dataclass"]
        direction TB
        
        subgraph Hardware["Hardware Limits"]
            MaxMem["max_memory_gb: 24.0"]
            ReserveMem["reserve_memory_gb: 4.0"]
            Threads["n_threads: 8"]
            Batch["n_batch: 2048"]
        end
        
        subgraph Model["Model Settings"]
            ModelId["model_id"]
            Ctx["n_ctx: 194560<br/>(192K context)"]
            Quant["quantization"]
            CacheK["cache_type_k: q4_0"]
            CacheV["cache_type_v: q4_0"]
        end
        
        subgraph GPU["GPU Offload"]
            NGPULayers["n_gpu_layers: 0"]
            MainGPU["main_gpu: 0"]
        end
        
        subgraph Features["Features"]
            FlashAttn["flash_attn: true"]
            MMap["use_mmap: true"]
            MLock["use_mlock: false<br/>(WSL2)"]
            Tools["enable_tools: true"]
            Vision["enable_vision: true"]
        end
    end

    subgraph MemMgr["💾 MemoryManager"]
        direction TB
        
        subgraph Monitor["Monitoring"]
            TotalRAM["TOTAL_RAM_GB = 28<br/>(GMKtech M7)"]
            GetStats["get_stats()<br/>Current usage"]
        end
        
        subgraph Calc["Calculations"]
            TargetUtil["target_utilization<br/>max_memory / total"]
            MoECalc["compute_moe_context()<br/>AirLLM formula"]
            SelectQuant["select_quantization()<br/>Based on params"]
        end
        
        subgraph Track["Tracking"]
            RegModel["register_model_memory()"]
            RegKV["register_kv_memory()"]
            CheckAvailable["check_available()"]
        end
    end

    subgraph Formulas["📐 Key Formulas"]
        direction TB
        
        MoEFormula["MoE Context Calculation:<br/>safe_ctx = compute_moe_context(<br/>model_disk_gb, n_layers,<br/>n_kv_heads, head_dim,<br/>kv_cache_type, is_moe)"]
        
        QuantFormula["Quantization Selection:<br/>&lt;7B → Q4_K_M<br/>7-20B → Q3_K_M<br/>&gt;20B → Q2_K<br/>&gt;100B MoE → UD-IQ2_XXS"]
        
        KVFormula["KV Cache Size:<br/>size = n_ctx × n_layers ×<br/>n_kv_heads × head_dim ×<br/>bytes_per_element"]
    end

    subgraph BackendCfg["🔧 Backend Config"]
        GetBackend["get_backend_config()<br/>• n_ctx, n_threads<br/>• cache_type_k/v<br/>• flash_attn<br/>• CPU mask/affinity"]
    end

    Default --> M7Config
    YAML --> M7Config
    Code --> M7Config
    Preset --> M7Config
    
    M7Config --> MemMgr
    M7Config --> BackendCfg
    
    Hardware --> TargetUtil
    Model --> MoECalc
    
    MemMgr --> Formulas
    
    style ConfigSrc fill:#e3f2fd
    style M7Config fill:#fff3e0
    style Hardware fill:#ffebee
    style Model fill:#e8f5e9
    style GPU fill:#fce4ec
    style Features fill:#f3e5f5
    style MemMgr fill:#e1f5fe
    style Formulas fill:#fff8e1
    style BackendCfg fill:#f5f5f5
```

## Memory Calculation Details

### MoE Context Sizing (AirLLM-inspired)

```python
# For MoE models, dynamically compute safe context from available RAM
# instead of hardcoded values

safe_ctx = memory_manager.compute_moe_context(
    model_disk_gb=disk_size,      # GGUF file size
    n_layers=62,                   # From architecture map
    n_kv_heads=8,
    head_dim=128,
    kv_cache_type="q4_0",         # May fallback from turbo3
    is_moe=True
)
```

### Quantization Bit-Depth Map

| Quantization | Bits | Use Case |
|--------------|------|----------|
| UD-TQ1_0 | 1.0 | Extreme compression |
| UD-IQ2_XXS | 2.06 | Large MoE on 28GB |
| Q2_K | 2.5 | Balanced small models |
| Q3_K_M | 3.4 | Quality/space tradeoff |
| Q4_K_M | 4.5 | **Recommended default** |
| Q8_0 | 8.0 | Maximum quality |

### KV Cache Compression

| Type | Bits | Reduction vs F16 |
|------|------|------------------|
| f16 | 16 | 1x (baseline) |
| q8_0 | 8 | 2x |
| q4_0 | 4 | 4x |
| turbo3 | ~3 | ~5x (falls back to q8_0) |

## Configuration Loading Priority

1. Explicit path: `load_config("path.yaml")`
2. Platform config dir: `%APPDATA%/miniforge/config.yaml` (Windows)
3. Fallback: `~/.config/miniforge/config.yaml`
4. Local: `./miniforge.yaml`
5. Local: `./config.yaml`
6. **Default M7Config()**
