param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsForward
)

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Manager = Join-Path $RootDir "model_manager.py"

if (-not $ArgsForward -or $ArgsForward.Count -eq 0) {
    & python $Manager status
} else {
    & python $Manager @ArgsForward
}
exit $LASTEXITCODE
