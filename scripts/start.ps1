#requires -Version 5.1
<#
.SYNOPSIS
    One-stop control script for the miniforge native-Windows LLM stack.

.DESCRIPTION
    Single entry point for every maintenance action: bootstrap the venv, install
    engine binaries, download GGUFs, launch the stack, stop it, show status, or
    run benchmarks. Replaces setup/stop/status/pull-models/obs-up/engines/*.ps1.

.PARAMETER Action
    What to do. Defaults to `setup` (build, don't start anything).

      setup              Create .venv, install deps, install llama.cpp Vulkan
      install-vulkan     (Re)install llama.cpp Vulkan binaries only
      install-ik-llama   Install ik_llama.cpp (download or build)
      pull               Download the REAP GGUF + Qwen3-30B control
      up                 Launch engine_manager + LiteLLM + Open WebUI
      down               Stop everything the stack started
      status             Live status of every service and port
      bench              Run the benchmark harness across engines
      obs-up             Launch Prometheus + Grafana (downloads if missing)
      all                setup -> install-vulkan -> up (skips pull; opt-in)

.PARAMETER WithAirLLM
    During `setup`, also install the [airllm] extra. Large.

.PARAMETER WithIkLlama
    During `setup`, also install the ik_llama.cpp binaries.

.PARAMETER Force
    Rebuild/redownload even if current version matches.

.PARAMETER Tag
    (install-vulkan) Pin a specific llama.cpp release tag like b8827.

.PARAMETER SkipVenv
    (setup) Reuse an existing .venv without recreating.

.PARAMETER SkipReap
    (pull) Skip the 56 GB MiniMax-REAP download.

.PARAMETER SkipQwen
    (pull) Skip the 18 GB Qwen3-30B-A3B download.

.PARAMETER ReapQuant
    (pull) Override REAP quant. Default Q2_K.

.PARAMETER ConvertIK
    (install-ik-llama) After installing, requantize Q2_K -> IQ2_KS. Slow.

.PARAMETER SourceModel
    (install-ik-llama -ConvertIK) Path to source GGUF for requantization.

.PARAMETER NoUI
    (up) Don't launch Open WebUI.

.PARAMETER NoLiteLLM
    (up) Don't launch LiteLLM; clients hit engine_manager directly.

.PARAMETER Watch
    (status) Refresh every N seconds.

.PARAMETER BenchAliases
    (bench) Model aliases to benchmark. Default: four REAP variants.

.PARAMETER BenchTarget
    (bench) OpenAI-compatible base URL. Default http://127.0.0.1:4000.

.PARAMETER BenchMaxPrompts
    (bench) Use only first N prompts. 0 = all.

.EXAMPLE
    .\scripts\start.ps1                        # default: setup only (no server start)
    .\scripts\start.ps1 -Action all            # setup + install Vulkan + launch stack
    .\scripts\start.ps1 -Action pull
    .\scripts\start.ps1 -Action up
    .\scripts\start.ps1 -Action status -Watch 5
    .\scripts\start.ps1 -Action bench -BenchAliases qwen3-30b-vulkan -BenchMaxPrompts 3
    .\scripts\start.ps1 -Action down
#>
param(
    [ValidateSet('setup', 'install-vulkan', 'install-ik-llama', 'pull', 'up', 'down', 'status', 'bench', 'obs-up', 'all')]
    [string]$Action = 'setup',

    # shared flags
    [switch]$Force,

    # setup flags
    [switch]$WithAirLLM,
    [switch]$WithIkLlama,
    [switch]$SkipVenv,

    # install-vulkan flags
    [string]$Tag,

    # pull flags
    [switch]$SkipReap,
    [switch]$SkipQwen,
    [string]$ReapQuant = 'Q2_K',
    [string]$ReapModelDir,
    [string]$ModelDir,

    # install-ik-llama flags
    [switch]$ConvertIK,
    [string]$SourceModel,
    [string]$TargetQuant = 'iq2_ks',

    # up flags
    [switch]$NoUI,
    [switch]$NoLiteLLM,
    [switch]$NoObs,

    # status flags
    [int]$Watch = 0,

    # bench flags
    [string[]]$BenchAliases = @(
        'minimax-reap-vulkan',
        'minimax-reap-ikllama',
        'minimax-reap-miniforge',
        'minimax-reap-airllm'
    ),
    [string]$BenchTarget = 'http://127.0.0.1:4000',
    [string]$BenchApiKey = 'sk-local-dev-key',
    [int]$BenchMaxPrompts = 0
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$LogDir    = Join-Path $RepoRoot 'logs'
$EnginesDir = Join-Path $RepoRoot 'engines'
$VenvPy    = Join-Path $RepoRoot '.venv\Scripts\python.exe'

# ===========================================================================
# shared helpers
# ===========================================================================

function Write-Section($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Ensure-Dir($p) {
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null }
}

function Load-DotEnv {
    $envFile = Join-Path $RepoRoot '.env'
    if (-not (Test-Path $envFile)) { return }
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*#') { return }
        if ($_ -match '^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.+?)\s*$') {
            $k = $Matches[1]
            $v = $Matches[2].Trim('"').Trim("'")
            [Environment]::SetEnvironmentVariable($k, $v, 'Process')
        }
    }
}

function Activate-Venv {
    $activate = Join-Path $RepoRoot '.venv\Scripts\Activate.ps1'
    if (Test-Path $activate) { . $activate }
}

function Get-PythonExe {
    if (Test-Path $VenvPy) { return $VenvPy }
    return 'python'
}

function Get-HfCommand {
    if (Get-Command 'hf' -ErrorAction SilentlyContinue)              { return 'hf' }
    if (Get-Command 'huggingface-cli' -ErrorAction SilentlyContinue) { return 'huggingface-cli' }
    throw "Neither 'hf' nor 'huggingface-cli' is available. Run `.\scripts\start.ps1 -Action setup` first."
}

function Github-Headers {
    $h = @{ 'User-Agent' = 'miniforge-installer' }
    if ($env:GITHUB_TOKEN) { $h['Authorization'] = "Bearer $env:GITHUB_TOKEN" }
    return $h
}

# ===========================================================================
# action: install-vulkan
# ===========================================================================

function Invoke-InstallVulkan {
    Write-Section "Installing llama.cpp Vulkan binaries"
    $engineDir   = Join-Path $EnginesDir 'llama-cpp-vulkan'
    $versionFile = Join-Path $engineDir 'VERSION'

    if ($Tag) {
        Write-Host "Fetching release $Tag..."
        $release = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$Tag" -Headers (Github-Headers)
    } else {
        Write-Host "Fetching latest llama.cpp release..."
        $release = Invoke-RestMethod 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest' -Headers (Github-Headers)
    }

    $tagName = $release.tag_name
    Write-Host "Release: $tagName"

    if ($tagName -match '^b(\d+)$' -and [int]$Matches[1] -lt 6000) {
        Write-Warning "Build $tagName is older than b6000; Vulkan iGPU detection may be broken. Use -Tag b6200 or newer."
    }

    if (-not $Force -and (Test-Path $versionFile)) {
        $current = (Get-Content $versionFile -Raw).Trim()
        if ($current -eq $tagName) {
            Write-Host "Already at $tagName. Use -Force to reinstall." -ForegroundColor Green
            return
        }
    }

    $asset = $release.assets | Where-Object { $_.name -match 'bin-win-vulkan-x64\.zip$' } | Select-Object -First 1
    if (-not $asset) { throw "No Vulkan x64 asset in release $tagName" }

    Write-Host "Downloading $($asset.name) ($([math]::Round($asset.size/1MB,1)) MB)..."
    $tmpZip = Join-Path $env:TEMP $asset.name
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $tmpZip

    if (Test-Path $engineDir) { Remove-Item -Recurse -Force $engineDir }
    Ensure-Dir $engineDir

    Write-Host "Extracting to $engineDir..."
    Expand-Archive -Path $tmpZip -DestinationPath $engineDir -Force
    Remove-Item $tmpZip -Force

    $nested = Join-Path $engineDir 'build\bin'
    if (Test-Path $nested) {
        Get-ChildItem $nested -Recurse | Move-Item -Destination $engineDir -Force
        Remove-Item (Join-Path $engineDir 'build') -Recurse -Force
    }

    Set-Content -Path $versionFile -Value $tagName -NoNewline
    $binary = Join-Path $engineDir 'llama-server.exe'
    if (-not (Test-Path $binary)) { throw "llama-server.exe not found at $binary" }

    Write-Host "Installed llama.cpp Vulkan $tagName" -ForegroundColor Green
    Write-Host "  Verify: .\engines\llama-cpp-vulkan\llama-server.exe --version"
    Write-Host "  List devices: .\engines\llama-cpp-vulkan\llama-server.exe --list-devices"
}

# ===========================================================================
# action: install-ik-llama
# ===========================================================================

function Invoke-InstallIkLlama {
    Write-Section "Installing ik_llama.cpp"
    $engineDir   = Join-Path $EnginesDir 'ik-llama'
    $versionFile = Join-Path $engineDir 'VERSION'
    $buildDir    = Join-Path $env:TEMP 'ik_llama_build'

    if (-not $Force -and (Test-Path (Join-Path $engineDir 'llama-server.exe'))) {
        Write-Host "ik_llama already installed. Use -Force to rebuild." -ForegroundColor Green
    } else {
        $prebuilt = $null
        try {
            $releases = Invoke-RestMethod 'https://api.github.com/repos/ikawrakow/ik_llama.cpp/releases' -Headers (Github-Headers)
            foreach ($r in $releases) {
                $a = $r.assets | Where-Object { $_.name -match 'win.*x64\.zip$' } | Select-Object -First 1
                if ($a) { $prebuilt = @{ Release = $r; Asset = $a }; break }
            }
        } catch {
            Write-Warning "Release check failed: $_"
        }

        if ($prebuilt) {
            $tmpZip = Join-Path $env:TEMP $prebuilt.Asset.name
            Write-Host "Downloading $($prebuilt.Asset.name)..."
            Invoke-WebRequest -Uri $prebuilt.Asset.browser_download_url -OutFile $tmpZip
            if (Test-Path $engineDir) { Remove-Item -Recurse -Force $engineDir }
            Ensure-Dir $engineDir
            Expand-Archive -Path $tmpZip -DestinationPath $engineDir -Force
            Remove-Item $tmpZip -Force
            foreach ($nested in @('build\bin\Release','build\bin','bin')) {
                $np = Join-Path $engineDir $nested
                if (Test-Path $np) {
                    Get-ChildItem $np -Recurse | Move-Item -Destination $engineDir -Force
                    $root = ($nested -split '\\')[0]
                    $rp = Join-Path $engineDir $root
                    if (Test-Path $rp) { Remove-Item $rp -Recurse -Force }
                    break
                }
            }
            Set-Content -Path $versionFile -Value $prebuilt.Release.tag_name -NoNewline
        } else {
            Write-Host "No prebuilt Windows binary found. Building from source." -ForegroundColor Yellow
            if (-not (Get-Command 'git' -ErrorAction SilentlyContinue))   { throw 'git not found. https://git-scm.com/download/win' }
            if (-not (Get-Command 'cmake' -ErrorAction SilentlyContinue)) { throw 'cmake not found. winget install Kitware.CMake' }
            $pf86 = [Environment]::GetFolderPath('ProgramFilesX86')
            $vswhere = Join-Path $pf86 'Microsoft Visual Studio\Installer\vswhere.exe'
            if (-not (Test-Path $vswhere)) {
                throw 'Visual Studio Build Tools not detected. Install "Build Tools for Visual Studio 2022" with "Desktop development with C++": https://visualstudio.microsoft.com/visual-cpp-build-tools/'
            }
            if (Test-Path $buildDir) { Remove-Item -Recurse -Force $buildDir }
            Write-Host "Cloning ik_llama.cpp..."
            git clone --depth 1 https://github.com/ikawrakow/ik_llama.cpp $buildDir 2>&1 | Out-Host
            Push-Location $buildDir
            try {
                $commit = (git rev-parse HEAD).Substring(0,7)
                Write-Host "Configuring (Release, AVX2 CPU)..."
                cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_AVX2=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF 2>&1 | Out-Host
                if ($LASTEXITCODE -ne 0) { throw 'cmake configure failed' }
                Write-Host "Building (10-20 min)..."
                cmake --build build --config Release --parallel 2>&1 | Out-Host
                if ($LASTEXITCODE -ne 0) { throw 'cmake build failed' }
                $binSrc = Join-Path $buildDir 'build\bin\Release'
                if (-not (Test-Path $binSrc)) { $binSrc = Join-Path $buildDir 'build\bin' }
                if (-not (Test-Path $binSrc)) { throw "Build output missing under $buildDir\build\bin" }
                if (Test-Path $engineDir) { Remove-Item -Recurse -Force $engineDir }
                Ensure-Dir $engineDir
                Get-ChildItem $binSrc -File | Copy-Item -Destination $engineDir -Force
                Set-Content -Path $versionFile -Value "git-$commit" -NoNewline
            } finally {
                Pop-Location
            }
        }
        Write-Host "Installed ik_llama.cpp" -ForegroundColor Green
    }

    if ($ConvertIK) {
        if (-not $SourceModel) { throw '-SourceModel required with -ConvertIK' }
        if (-not (Test-Path $SourceModel)) { throw "Source GGUF not found: $SourceModel" }
        $qbin = Join-Path $engineDir 'llama-quantize.exe'
        if (-not (Test-Path $qbin)) { throw 'llama-quantize.exe missing; required for -ConvertIK' }
        $srcItem = Get-Item $SourceModel
        $dstName = $srcItem.BaseName -replace 'Q2_K', $TargetQuant.ToUpper()
        $dstPath = Join-Path $srcItem.DirectoryName "$dstName.gguf"
        Write-Host "Requantizing $SourceModel -> $dstPath ($TargetQuant) (20-60 min)..."
        & $qbin $SourceModel $dstPath $TargetQuant
        if ($LASTEXITCODE -ne 0) { throw "llama-quantize exited with $LASTEXITCODE" }
        Write-Host "Wrote $dstPath" -ForegroundColor Green
    }
}

# ===========================================================================
# action: setup
# ===========================================================================

function Invoke-Setup {
    Write-Section "miniforge native-Windows bootstrap"

    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        throw 'python not on PATH. Install Python 3.11+ from https://www.python.org/downloads/ and restart your shell.'
    }
    $pyVer = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ([version]$pyVer -lt [version]'3.11') { throw "Python $pyVer is too old. Need 3.11+." }
    Write-Host "python $pyVer OK"

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "Installing uv via pip..." -ForegroundColor Yellow
        & python -m pip install --user --upgrade uv
        if ($LASTEXITCODE -ne 0) { throw 'pip install uv failed' }
    }
    Write-Host "uv OK"

    if (-not $SkipVenv) {
        if (-not (Test-Path '.venv')) {
            Write-Host "Creating .venv..."
            & uv venv --python "$pyVer"
            if ($LASTEXITCODE -ne 0) { throw 'uv venv failed' }
        } else {
            Write-Host ".venv exists, skipping (use -SkipVenv`:`$false after deleting it to recreate)"
        }
    }

    Write-Host "Installing miniforge[all]..."
    & uv pip install -e ".[all]" --index-strategy unsafe-best-match
    if ($LASTEXITCODE -ne 0) { throw 'uv pip install .[all] failed' }

    if ($WithAirLLM) {
        Write-Host "Installing [airllm]..."
        & uv pip install -e ".[airllm]" --index-strategy unsafe-best-match
        if ($LASTEXITCODE -ne 0) { Write-Warning '[airllm] install failed; AirLLM engine will be unavailable.' }
    }

    Invoke-InstallVulkan

    # Isolated uv tools — open-webui and litellm pin conflicting httpx/uvicorn
    # versions; keep them off the project's dep graph.
    Write-Host "Installing Open WebUI as isolated uv tool..."
    & uv tool install open-webui
    if ($LASTEXITCODE -ne 0) { Write-Warning 'uv tool install open-webui failed; -Action up will skip the UI.' }

    Write-Host "Installing LiteLLM proxy as isolated uv tool..."
    & uv tool install 'litellm[proxy]'
    if ($LASTEXITCODE -ne 0) { Write-Warning 'uv tool install litellm failed; -Action up will skip LiteLLM.' }

    if ($WithIkLlama) { Invoke-InstallIkLlama }

    foreach ($d in @('bench\results', 'data\openwebui', 'engines', 'models', 'logs')) {
        Ensure-Dir (Join-Path $RepoRoot $d)
    }

    $envExample = Join-Path $RepoRoot '.env.example'
    $envFile    = Join-Path $RepoRoot '.env'
    if ((Test-Path $envExample) -and (-not (Test-Path $envFile))) {
        Copy-Item $envExample $envFile
        Write-Host ""
        Write-Host "Created .env from .env.example. Edit HF_TOKEN / MODEL_DIR / MODEL_DIR_REAP." -ForegroundColor Yellow
    }

    Write-Section "Setup complete"
    Write-Host "Next:"
    Write-Host "  .\scripts\start.ps1 -Action pull        # download REAP GGUF (~56 GB)"
    Write-Host "  .\scripts\start.ps1 -Action up          # launch the stack"
    Write-Host "  .\scripts\start.ps1 -Action all         # setup + up together"
}

# ===========================================================================
# action: pull
# ===========================================================================

function Invoke-Pull {
    Write-Section "Downloading model GGUFs"
    Activate-Venv

    if (-not $ReapModelDir) { $ReapModelDir = $env:MODEL_DIR_REAP }
    if (-not $ReapModelDir) { $ReapModelDir = 'D:\ai\models' }
    if (-not $ModelDir)     { $ModelDir     = $env:MODEL_DIR }
    if (-not $ModelDir)     { $ModelDir     = 'C:\Users\midwe\miniforge\gguf' }

    Ensure-Dir $ReapModelDir
    Ensure-Dir $ModelDir

    $hf = Get-HfCommand
    function Pull-Repo($repo, $pattern, $localDir) {
        Write-Host ""
        Write-Host "==> $repo" -ForegroundColor Cyan
        Write-Host "    filter : $pattern"
        Write-Host "    target : $localDir"
        $args = @('download', $repo, '--include', $pattern, '--local-dir', $localDir)
        if ($env:HF_TOKEN) { $args += @('--token', $env:HF_TOKEN) }
        & $hf @args
        if ($LASTEXITCODE -ne 0) { throw "$hf exited with $LASTEXITCODE" }
    }

    if (-not $SkipReap) {
        $target = Join-Path $ReapModelDir 'MiniMax-M2.7-161B-REAP-GGUF'
        Write-Host "Pulling MiniMax-M2.7-161B-REAP $ReapQuant (~56 GB for Q2_K)." -ForegroundColor Yellow
        Pull-Repo '0xSero/MiniMax-M2.7-161B-REAP-GGUF' "*$ReapQuant*.gguf" $target
    } else {
        Write-Host "Skipping REAP (-SkipReap)." -ForegroundColor DarkGray
    }

    if (-not $SkipQwen) {
        $target = Join-Path $ModelDir 'Qwen3-30B-A3B-Instruct-GGUF'
        Write-Host "Pulling Qwen3-30B-A3B UD-Q4_K_M (~18 GB)." -ForegroundColor Yellow
        Pull-Repo 'unsloth/Qwen3-30B-A3B-Instruct-GGUF' '*UD-Q4_K_M*.gguf' $target
    } else {
        Write-Host "Skipping Qwen3-30B (-SkipQwen)." -ForegroundColor DarkGray
    }

    Write-Host ""
    Write-Host "Done." -ForegroundColor Green
    Write-Host "Existing MiniMax-M2.7 UD-IQ2_XXS kept in C:\Users\midwe\miniforge\gguf\UD-IQ2_XXS\"
    Write-Host "Env hints for .env:"
    Write-Host "  MODEL_DIR=$ModelDir"
    Write-Host "  MODEL_DIR_REAP=$ReapModelDir"
}

# ===========================================================================
# action: up
# ===========================================================================

function Start-Service-Background {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$Arguments,
        [string]$WorkingDirectory = $RepoRoot
    )
    $logPath = Join-Path $LogDir "$Name.log"
    $pidPath = Join-Path $LogDir "$Name.pid"
    if (Test-Path $pidPath) {
        $existing = Get-Content $pidPath -ErrorAction SilentlyContinue
        if ($existing -and (Get-Process -Id $existing -ErrorAction SilentlyContinue)) {
            Write-Warning "$Name already running (pid $existing). Run -Action down first."
            return
        }
    }
    Write-Host "Starting $Name..." -ForegroundColor Cyan
    $proc = Start-Process -FilePath $FilePath `
        -ArgumentList $Arguments `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $logPath `
        -RedirectStandardError "$logPath.err" `
        -WindowStyle Hidden `
        -PassThru
    Set-Content -Path $pidPath -Value $proc.Id -NoNewline
    Write-Host "  $Name pid=$($proc.Id), log=$logPath"
}

function Invoke-Up {
    Write-Section "Launching miniforge stack"
    Activate-Venv
    Load-DotEnv
    Ensure-Dir $LogDir

    # LiteLLM + Open WebUI print Unicode banners via click/print; cp1252 stderr/stdout aborts before bind.
    [Environment]::SetEnvironmentVariable('PYTHONUTF8', '1', 'Process')
    [Environment]::SetEnvironmentVariable('PYTHONIOENCODING', 'utf-8', 'Process')

    $python = Get-PythonExe

    Start-Service-Background -Name 'engine_manager' -FilePath $python `
        -Arguments @('-u', '-m', 'miniforge.launcher.engine_manager', '--config', 'config/engines.yaml')

    if (-not $NoLiteLLM) {
        $master = if ($env:LITELLM_MASTER_KEY) { $env:LITELLM_MASTER_KEY } else { 'sk-local-dev-key' }
        $env:LITELLM_MASTER_KEY = $master
        $uvExe = (Get-Command uv -ErrorAction Stop).Source
        Start-Service-Background -Name 'litellm' -FilePath $uvExe `
            -Arguments @('tool', 'run', '--from', 'litellm[proxy]', 'litellm',
                         '--config', 'config/litellm.yaml',
                         '--host', '127.0.0.1', '--port', '4000')
    }

    if (-not $NoUI) {
        $owuiEnv = Join-Path $RepoRoot 'config\openwebui.env'
        if (Test-Path $owuiEnv) {
            Get-Content $owuiEnv | ForEach-Object {
                if ($_ -match '^\s*#' -or $_ -notmatch '=') { return }
                $k, $v = $_ -split '=', 2
                [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), 'Process')
            }
        }
        if (-not $NoLiteLLM) {
            $owKey = if ($env:LITELLM_MASTER_KEY) { $env:LITELLM_MASTER_KEY } else { 'sk-local-dev-key' }
            [Environment]::SetEnvironmentVariable('OPENAI_API_KEY', $owKey, 'Process')
        }
        if (-not $uvExe) { $uvExe = (Get-Command uv -ErrorAction Stop).Source }
        Start-Service-Background -Name 'openwebui' -FilePath $uvExe `
            -Arguments @('tool', 'run', 'open-webui',
                         'serve', '--host', '127.0.0.1', '--port', '8080')
    }

    if (-not $NoObs) {
        Start-Obs-Services
    }

    Write-Host ""
    Write-Host "Stack starting. URLs:" -ForegroundColor Green
    Write-Host "  Open WebUI      http://127.0.0.1:8080"
    Write-Host "  LiteLLM proxy   http://127.0.0.1:4000"
    Write-Host "  engine_manager  http://127.0.0.1:9292"
    if (-not $NoObs) {
        Write-Host "  Prometheus      http://127.0.0.1:9090"
        Write-Host "  Grafana         http://127.0.0.1:3000  (admin/admin)"
    }
    Write-Host ""
    Write-Host "Tail logs:   Get-Content -Wait .\logs\engine_manager.log"
    Write-Host "Status:      .\scripts\start.ps1 -Action status"
    Write-Host "Stop:        .\scripts\start.ps1 -Action down"
}

# ===========================================================================
# action: down
# ===========================================================================

function Stop-ByPidFile($name) {
    $pidFile = Join-Path $LogDir "$name.pid"
    if (-not (Test-Path $pidFile)) { return }
    # $PID is reserved (current process); use a different name.
    $procId = (Get-Content $pidFile -Raw).Trim()
    if ($procId -and ($procId -as [int])) {
        Write-Host "Stopping $name (pid $procId)..." -ForegroundColor Yellow
        try { Stop-Process -Id ([int]$procId) -Force -ErrorAction Stop }
        catch { Write-Host "  already gone" }
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

function Invoke-Down {
    Write-Section "Stopping miniforge stack"
    Ensure-Dir $LogDir

    Stop-ByPidFile 'grafana'
    Stop-ByPidFile 'prometheus'
    Stop-ByPidFile 'openwebui'
    Stop-ByPidFile 'litellm'
    Stop-ByPidFile 'engine_manager'

    foreach ($name in @('llama-server','open-webui','open_webui','litellm','prometheus','grafana-server')) {
        Get-Process -Name $name -ErrorAction SilentlyContinue | ForEach-Object {
            try { Stop-Process -Id $_.Id -Force -ErrorAction Stop } catch {}
        }
    }

    foreach ($p in @(8001, 8002, 8003, 8004, 4000, 8080, 9292, 9090, 3000)) {
        $conns = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
        foreach ($c in $conns) {
            if ($c.OwningProcess -gt 4) {
                try { Stop-Process -Id $c.OwningProcess -Force -ErrorAction Stop } catch {}
            }
        }
    }

    Write-Host "Stopped." -ForegroundColor Green
}

# ===========================================================================
# action: status
# ===========================================================================

function Invoke-Status {
    $services = @(
        @{ Name = 'engine_manager';   Port = 9292; Health = '/health' },
        @{ Name = 'litellm';          Port = 4000; Health = '/health/readiness' },
        @{ Name = 'openwebui';        Port = 8080; Health = '/health' },
        @{ Name = 'prometheus';       Port = 9090; Health = '/-/ready' },
        @{ Name = 'grafana';          Port = 3000; Health = '/api/health' },
        @{ Name = 'miniforge-server'; Port = 8003; Health = '/health' },
        @{ Name = 'miniforge-airllm'; Port = 8004; Health = '/health' },
        @{ Name = 'llama_vulkan';     Port = 8001; Health = '/health' },
        @{ Name = 'ik_llama';         Port = 8002; Health = '/health' }
    )

    function Show-Once {
        $now = Get-Date -Format 'HH:mm:ss'
        Write-Host ""
        Write-Host "miniforge status @ $now" -ForegroundColor Cyan
        Write-Host ("{0,-22} {1,-10} {2,-10} {3}" -f 'service','port','listening','details')
        Write-Host ('-' * 80)
        foreach ($s in $services) {
            $listen = Get-NetTCPConnection -LocalPort $s.Port -State Listen -ErrorAction SilentlyContinue
            $isUp = [bool]$listen
            $detail = ""
            if ($isUp) {
                try {
                    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$($s.Port)$($s.Health)" -TimeoutSec 2 -ErrorAction Stop
                    $detail = ($resp | ConvertTo-Json -Compress -Depth 3)
                    if ($detail.Length -gt 60) { $detail = $detail.Substring(0, 57) + '...' }
                } catch { $detail = "(no /health)" }
            }
            $color  = if ($isUp) { 'Green' } else { 'DarkGray' }
            $upText = if ($isUp) { 'yes' } else { 'no' }
            Write-Host ("{0,-22} {1,-10} {2,-10} {3}" -f $s.Name, $s.Port, $upText, $detail) -ForegroundColor $color
        }
        try {
            $mx = Invoke-RestMethod -Uri 'http://127.0.0.1:9292/engine' -TimeoutSec 2 -ErrorAction Stop
            Write-Host ""
            Write-Host "active engine alias: $($mx.alias)" -ForegroundColor Yellow
        } catch {}
        try {
            $metrics = Invoke-WebRequest -Uri 'http://127.0.0.1:9292/metrics' -TimeoutSec 2 -UseBasicParsing
            $tpsLine = ($metrics.Content -split "`n") | Where-Object { $_ -match '^miniforge_tps ' } | Select-Object -First 1
            if ($tpsLine) { Write-Host "last generation TPS: $(($tpsLine -split ' ')[1])" -ForegroundColor Yellow }
        } catch {}
    }

    if ($Watch -le 0) { Show-Once; return }
    while ($true) { Clear-Host; Show-Once; Start-Sleep -Seconds $Watch }
}

# ===========================================================================
# action: bench
# ===========================================================================

function Invoke-Bench {
    Write-Section "Running benchmark sweep"
    Activate-Venv

    $resultsDir = Join-Path $RepoRoot 'bench\results'
    $promptsFile = Join-Path $RepoRoot 'bench\prompts.jsonl'
    if (-not (Test-Path $promptsFile)) { throw "Missing $promptsFile" }
    Ensure-Dir $resultsDir

    $python = Get-PythonExe
    $stamp  = Get-Date -Format 'yyyyMMdd-HHmmss'

    foreach ($alias in $BenchAliases) {
        $out = Join-Path $resultsDir "$alias-$stamp.json"
        Write-Host ""
        Write-Host "==> $alias" -ForegroundColor Cyan
        $pyArgs = @(
            (Join-Path $RepoRoot 'bench\runner.py'),
            '--alias', $alias,
            '--target', $BenchTarget,
            '--api-key', $BenchApiKey,
            '--prompts', $promptsFile,
            '--output', $out
        )
        if ($BenchMaxPrompts -gt 0) { $pyArgs += @('--max-prompts', $BenchMaxPrompts) }
        & $python @pyArgs
        if ($LASTEXITCODE -ne 0) { Write-Warning "$alias bench failed (exit $LASTEXITCODE). Continuing." }
    }

    Write-Host ""
    Write-Host "==> Compiling SUMMARY.md" -ForegroundColor Cyan
    & $python (Join-Path $RepoRoot 'bench\compare.py') $resultsDir --out (Join-Path $resultsDir 'SUMMARY.md')
}

# ===========================================================================
# action: obs-up
# ===========================================================================

function Get-ObsPaths {
    $obsDir  = Join-Path $EnginesDir '_obs'
    return @{
        ObsDir  = $obsDir
        PromDir = Join-Path $obsDir 'prometheus'
        GrafDir = Join-Path $obsDir 'grafana'
    }
}

function Ensure-Obs-Installed {
    $paths = Get-ObsPaths
    Ensure-Dir $paths.ObsDir

    if ($Force -or -not (Test-Path (Join-Path $paths.PromDir 'prometheus.exe'))) {
        Write-Host "Downloading Prometheus..."
        $release = Invoke-RestMethod 'https://api.github.com/repos/prometheus/prometheus/releases/latest' -Headers (Github-Headers)
        $asset = $release.assets | Where-Object { $_.name -match 'windows-amd64\.zip$' } | Select-Object -First 1
        if (-not $asset) { throw 'no windows-amd64 prometheus asset' }
        $zip = Join-Path $env:TEMP $asset.name
        Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip
        if (Test-Path $paths.PromDir) { Remove-Item -Recurse -Force $paths.PromDir }
        Expand-Archive -Path $zip -DestinationPath $paths.ObsDir -Force
        $ex = Get-ChildItem $paths.ObsDir -Directory | Where-Object Name -Like 'prometheus-*' | Select-Object -First 1
        if ($ex) { Rename-Item -Path $ex.FullName -NewName 'prometheus' }
        Remove-Item $zip -Force
    }

    if ($Force -or -not (Test-Path (Join-Path $paths.GrafDir 'bin\grafana-server.exe'))) {
        Write-Host "Downloading Grafana OSS..."
        $grafUrl = 'https://dl.grafana.com/oss/release/grafana-11.3.0.windows-amd64.zip'
        $zip = Join-Path $env:TEMP 'grafana.zip'
        Invoke-WebRequest -Uri $grafUrl -OutFile $zip
        if (Test-Path $paths.GrafDir) { Remove-Item -Recurse -Force $paths.GrafDir }
        Expand-Archive -Path $zip -DestinationPath $paths.ObsDir -Force
        $ex = Get-ChildItem $paths.ObsDir -Directory | Where-Object Name -Like 'grafana-*' | Select-Object -First 1
        if ($ex) { Rename-Item -Path $ex.FullName -NewName 'grafana' }
        Remove-Item $zip -Force
    }

    $provDir     = Join-Path $paths.GrafDir 'conf\provisioning'
    $dashProvDir = Join-Path $provDir 'dashboards'
    $dsProvDir   = Join-Path $provDir 'datasources'
    Ensure-Dir $dashProvDir
    Ensure-Dir $dsProvDir

    $dsYaml = @"
apiVersion: 1
datasources:
  - name: prometheus
    type: prometheus
    access: proxy
    url: http://127.0.0.1:9090
    uid: prometheus
    isDefault: true
"@
    Set-Content -Path (Join-Path $dsProvDir 'miniforge.yaml') -Value $dsYaml -Encoding UTF8

    $dashesRoot = Join-Path $paths.GrafDir 'data\dashboards\miniforge'
    Ensure-Dir $dashesRoot
    $dashSrc = Join-Path $RepoRoot 'config\grafana\dashboards\llm-engines.json'
    if (Test-Path $dashSrc) {
        Copy-Item -Force $dashSrc -Destination $dashesRoot
    }

    $dashYaml = @"
apiVersion: 1
providers:
  - name: miniforge
    folder: miniforge
    type: file
    options:
      path: $dashesRoot
"@
    Set-Content -Path (Join-Path $dashProvDir 'miniforge.yaml') -Value $dashYaml -Encoding UTF8
}

function Start-Obs-Services {
    $paths   = Get-ObsPaths
    $promExe = Join-Path $paths.PromDir 'prometheus.exe'
    $grafExe = Join-Path $paths.GrafDir 'bin\grafana-server.exe'

    if (-not ((Test-Path $promExe) -and (Test-Path $grafExe))) {
        Ensure-Obs-Installed
    }

    $promCfg = Join-Path $RepoRoot 'config\prometheus.yml'
    Start-Service-Background -Name 'prometheus' -FilePath $promExe `
        -Arguments @("--config.file=$promCfg",
                     '--web.listen-address=127.0.0.1:9090',
                     "--storage.tsdb.path=$($paths.ObsDir)\prometheus-data") `
        -WorkingDirectory $paths.PromDir

    Start-Service-Background -Name 'grafana' -FilePath $grafExe `
        -Arguments @("--homepath=$($paths.GrafDir)") `
        -WorkingDirectory $paths.GrafDir
}

function Invoke-ObsUp {
    Write-Section "Launching Prometheus + Grafana"
    Ensure-Dir $LogDir
    Ensure-Obs-Installed
    Start-Obs-Services
    Write-Host "Prometheus http://127.0.0.1:9090" -ForegroundColor Green
    Write-Host "Grafana    http://127.0.0.1:3000 (admin/admin)" -ForegroundColor Green
}

# ===========================================================================
# dispatch
# ===========================================================================

switch ($Action) {
    'setup'            { Invoke-Setup }
    'install-vulkan'   { Invoke-InstallVulkan }
    'install-ik-llama' { Invoke-InstallIkLlama }
    'pull'             { Invoke-Pull }
    'up'               { Invoke-Up }
    'down'             { Invoke-Down }
    'status'           { Invoke-Status }
    'bench'            { Invoke-Bench }
    'obs-up'           { Invoke-ObsUp }
    'all'              {
        Invoke-Setup
        Invoke-Up
    }
    default            { throw "Unknown action: $Action" }
}
