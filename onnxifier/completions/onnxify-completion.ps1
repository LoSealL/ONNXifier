# PowerShell argument completion for onnxify
# To enable, dot-source this script in your PowerShell session:
#   . /path/to/onnxify-completion.ps1
#
# To make it permanent, add the above line to your $PROFILE.
# Or run the helper function below after dot-sourcing:
#   Install-OnnxifyCompletion

$script:_onnxifyPassesCache = $null
$script:_onnxifyCacheTimestamp = $null

function Get-OnnxifyPassList {
    param([string]$Prefix = "")

    $cacheDuration = New-TimeSpan -Minutes 60
    $now = Get-Date

    if ($null -eq $script:_onnxifyCacheTimestamp -or ($now - $script:_onnxifyCacheTimestamp) -gt $cacheDuration) {
        $pythonCmd = if ($env:ONNXIFY_PYTHON) { $env:ONNXIFY_PYTHON } else { "python" }
        try {
            $output = & $pythonCmd -c "from onnxifier.passes import PASSES; print('\n'.join(PASSES))" 2>$null
            if ($LASTEXITCODE -eq 0 -and $output) {
                $script:_onnxifyPassesCache = $output -split "`r?`n" | Where-Object { $_ }
                $script:_onnxifyCacheTimestamp = $now
            }
        }
        catch {
            # Silently fail so the user still gets default completion
        }
    }

    if ($Prefix) {
        return $script:_onnxifyPassesCache | Where-Object { $_ -like "$Prefix*" }
    }
    return $script:_onnxifyPassesCache
}

$onnxifyCompletionScript = {
    param($wordToComplete, $commandAst, $cursorPosition)

    $tokenTexts = foreach ($t in $commandAst.CommandElements) { $t.Extent.Text }

    # Look backward to find the most recent option flag.
    # This handles space-separated multi-pass inputs like:
    #   onnxify model.onnx -a infer_shape fold_const<TAB>
    $lastOpt = $null
    for ($i = $tokenTexts.Count - 1; $i -ge 0; $i--) {
        if ($tokenTexts[$i] -ne $wordToComplete -and $tokenTexts[$i] -match '^-') {
            $lastOpt = $tokenTexts[$i]
            break
        }
    }

    switch -Regex ($lastOpt) {
        '^(-a|--activate|-r|--remove)$' {
            # If the current word looks like a new option (and is not empty),
            # fall through to option completion instead of pass completion.
            if ($wordToComplete -notmatch '^-' -or $wordToComplete -eq '') {
                $prefix = $wordToComplete
                $base = ""
                if ($wordToComplete -match '^(.*,)([^,]*)$') {
                    $base = $Matches[1]
                    $prefix = $Matches[2]
                }

                $passes = Get-OnnxifyPassList -Prefix $prefix
                foreach ($p in $passes) {
                    "$base$p"
                }
                return
            }
        }
    }

    # Specific option value completions based on immediate predecessor
    $prevToken = $null
    for ($i = $commandAst.CommandElements.Count - 1; $i -ge 0; $i--) {
        $text = $commandAst.CommandElements[$i].Extent.Text
        if ($text -ne $wordToComplete -and $text -notmatch '^\s*$') {
            $prevToken = $text
            break
        }
    }

    switch -Regex ($prevToken) {
        '^--format$' {
            @('protobuf', 'textproto', 'json', 'onnxtxt') | Where-Object { $_ -like "$wordToComplete*" }
            return
        }
        '^--checker-backend$' {
            @('onnx', 'openvino', 'onnxruntime') | Where-Object { $_ -like "$wordToComplete*" }
            return
        }
        '^(-vv|--log-level)$' {
            @('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL') | Where-Object { $_ -like "$wordToComplete*" }
            return
        }
        '^--print$' {
            $passes = Get-OnnxifyPassList -Prefix $wordToComplete
            $allMatches = @('all', 'l1', 'l2', 'l3') + $passes
            $allMatches | Where-Object { $_ -like "$wordToComplete*" } | Select-Object -Unique
            return
        }
    }

    if ($wordToComplete -match '^-') {
        $opts = @(
            '-a', '--activate',
            '-r', '--remove',
            '-n', '--no-passes',
            '--print',
            '--format',
            '-s', '--infer-shapes',
            '-c', '--config-file',
            '-u', '--uncheck',
            '--check',
            '-d', '--dry-run',
            '--checker-backend',
            '-v', '--opset-version',
            '-vv', '--log-level',
            '-R', '--recursive',
            '--nodes',
            '-h', '--help'
        )
        $opts | Where-Object { $_ -like "$wordToComplete*" }
    }
}

Register-ArgumentCompleter -CommandName onnxify -ScriptBlock $onnxifyCompletionScript

function Install-OnnxifyCompletion {
    <#
    .SYNOPSIS
        Persist the onnxify tab-completion by adding it to your PowerShell profile.
    .DESCRIPTION
        This function appends a dot-source line to your $PROFILE (CurrentUserCurrentHost).
        If the profile file or its parent directory does not exist, they are created.
    #>
    [CmdletBinding()]
    param()

    $profilePath = $PROFILE
    if ([string]::IsNullOrWhiteSpace($profilePath)) {
        Write-Error 'PowerShell `$PROFILE` is not defined in this host. Cannot install completion.'
        return
    }

    $profileDir = Split-Path -Parent $profilePath
    if (-not (Test-Path $profileDir)) {
        New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
        Write-Host "Created profile directory: $profileDir" -ForegroundColor Green
    }

    $scriptPath = $PSCommandPath
    if (-not $scriptPath) {
        Write-Error 'Unable to determine the path of the completion script. Make sure you dot-source the script first ( . /path/to/onnxify-completion.ps1 ).'
        return
    }

    $sourceLine = ". `"$scriptPath`""

    if (Test-Path $profilePath) {
        $existing = Get-Content -Raw $profilePath -ErrorAction SilentlyContinue
        if ($existing -and $existing.Contains($scriptPath)) {
            Write-Host 'Completion script is already referenced in your profile. No changes made.' -ForegroundColor Cyan
            return
        }
    }

    Add-Content -Path $profilePath -Value "`n# onnxify tab-completion`n$sourceLine`n"
    Write-Host "Installed onnxify completion to: $profilePath" -ForegroundColor Green
    Write-Host 'Restart your PowerShell session (or run . $PROFILE) to activate.' -ForegroundColor Yellow
}
