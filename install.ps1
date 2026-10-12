param(
    [string]$PythonPath = "",
    [string]$VenvPath = ".venv-rag"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ProjectRoot

function Invoke-WingetInstall {
    param(
        [string]$PackageId,
        [string[]]$ExtraArguments = @()
    )

    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "winget is required to install machine-level prerequisites. Install App Installer, then rerun this script."
    }

    & $winget.Source install --id $PackageId --exact --silent `
        --accept-source-agreements --accept-package-agreements @ExtraArguments
    if ($LASTEXITCODE -ne 0) {
        throw "winget could not install $PackageId (exit code $LASTEXITCODE)."
    }
}

function Resolve-Python312 {
    if ($PythonPath) {
        if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
            throw "Python executable not found: $PythonPath"
        }
        return (Resolve-Path -LiteralPath $PythonPath).Path
    }

    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        $resolved = & $launcher.Source -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved) {
            return ($resolved | Select-Object -Last 1).Trim()
        }
    }

    return $null
}

function Find-Ollama {
    $command = Get-Command ollama.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"),
        (Join-Path $env:ProgramFiles "Ollama\ollama.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    return $null
}

function Test-OllamaServer {
    try {
        $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3
        return $true
    }
    catch {
        return $false
    }
}

function Test-VCTools {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
        return $false
    }

    $installation = & $vswhere -latest -products "*" `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    return ($LASTEXITCODE -eq 0 -and [bool]$installation)
}

try {
    $basePython = Resolve-Python312
    if (-not $basePython) {
        Write-Host "Installing Python 3.12..."
        Invoke-WingetInstall -PackageId "Python.Python.3.12"
        $basePython = Resolve-Python312
        if (-not $basePython) {
            throw "Python 3.12 was installed, but the py launcher cannot find it in this terminal. Open a new PowerShell window and rerun install.ps1, or pass -PythonPath with the Python 3.12 executable."
        }
    }

    $pythonVersion = & $basePython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0 -or $pythonVersion.Trim() -ne "3.12") {
        throw "Python 3.12 is required. Found '$($pythonVersion.Trim())' at $basePython."
    }
    Write-Host "Using Python $($pythonVersion.Trim()): $basePython"

    if (-not (Test-VCTools)) {
        Write-Host "Installing Visual C++ Build Tools required by Chroma on Windows..."
        Invoke-WingetInstall -PackageId "Microsoft.VisualStudio.2022.BuildTools" -ExtraArguments @(
            "--override",
            "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
        )
        if (-not (Test-VCTools)) {
            throw "Visual C++ Build Tools were installed but are not available yet. Restart Windows if requested, then rerun install.ps1."
        }
    }

    $ollama = Find-Ollama
    if (-not $ollama) {
        Write-Host "Installing Ollama..."
        Invoke-WingetInstall -PackageId "Ollama.Ollama"
        $ollama = Find-Ollama
        if (-not $ollama) {
            throw "Ollama was installed but its executable is not available in this terminal. Open a new PowerShell window and rerun install.ps1."
        }
    }

    $resolvedVenv = if ([System.IO.Path]::IsPathRooted($VenvPath)) {
        $VenvPath
    }
    else {
        Join-Path $ProjectRoot $VenvPath
    }
    $venvPython = Join-Path $resolvedVenv "Scripts\python.exe"

    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        $venvVersion = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($LASTEXITCODE -ne 0 -or $venvVersion.Trim() -ne "3.12") {
            throw "The selected environment '$resolvedVenv' is not Python 3.12. Choose another -VenvPath; the existing environment was left unchanged."
        }
    }
    elseif (Test-Path -LiteralPath $resolvedVenv) {
        throw "The selected environment path exists but is not a Python virtual environment: $resolvedVenv"
    }
    else {
        & $basePython -m venv $resolvedVenv
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create the virtual environment at $resolvedVenv."
        }
    }

    Write-Host "Installing Python packages into $resolvedVenv..."
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Could not upgrade pip in $resolvedVenv."
    }
    & $venvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install the Python dependencies."
    }

    if (-not (Test-OllamaServer)) {
        Write-Host "Starting the local Ollama service..."
        $null = Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden -PassThru
        $serverReady = $false
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            Start-Sleep -Seconds 2
            if (Test-OllamaServer) {
                $serverReady = $true
                break
            }
        }
        if (-not $serverReady) {
            throw "Ollama did not become available at http://localhost:11434. Start Ollama and rerun install.ps1."
        }
    }

    $models = & $venvPython -c "import config; print(config.EMBEDDING_MODEL); print(config.LLM_MODEL)"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read the model names from config.py."
    }
    foreach ($model in $models) {
        $model = $model.Trim()
        if ($model) {
            Write-Host "Ensuring Ollama model is installed: $model"
            & $ollama pull $model
            if ($LASTEXITCODE -ne 0) {
                throw "Could not download Ollama model '$model'."
            }
        }
    }

    $pdfs = @(Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "documents") `
        -Filter "*.pdf" -File -ErrorAction SilentlyContinue)
    $indexFile = Join-Path $ProjectRoot "chroma_db\chroma.sqlite3"
    if ($pdfs.Count -gt 0 -and -not (Test-Path -LiteralPath $indexFile)) {
        Write-Host "Indexing PDFs found in documents..."
        & $venvPython (Join-Path $ProjectRoot "ingest.py")
        if ($LASTEXITCODE -ne 0) {
            throw "PDF ingestion failed. Resolve the error and rerun install.ps1."
        }
    }
    elseif ($pdfs.Count -eq 0) {
        Write-Warning "No PDFs found in documents. Add PDFs and run: `"$venvPython`" .\ingest.py"
    }
    else {
        Write-Host "Existing Chroma index found; leaving it unchanged."
    }

    Write-Host ""
    Write-Host "Installation steps completed."
    Write-Host "Run the post-install checks:"
    Write-Host "  `"$venvPython`" .\check_setup.py"
    Write-Host "Then run the sample RAG system tests:"
    Write-Host "  `"$venvPython`" .\Test\systemtest.py"
    Write-Host "Then start the app:"
    Write-Host "  `"$venvPython`" .\chat.py"
}
finally {
    Pop-Location
}
