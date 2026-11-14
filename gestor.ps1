param(
    [int]$Port = 8765,
    [switch]$NoBrowser
)

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Manager = Join-Path $RootDir "model_manager.py"

$CmdArgs = @("ui", "--port", "$Port")
if ($NoBrowser) {
    $CmdArgs += "--no-browser"
}

& python $Manager @CmdArgs
exit $LASTEXITCODE
