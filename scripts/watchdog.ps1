# Keeps the experiment jobs listed in results\jobs.txt running until each is complete.
#
# jobs.txt, one job per line (';'-separated, '#' starts a comment line):
#   name ; python args ; log file ; count file ; line regex ; expected count ; after (comma-separated job names)
# A job is complete when the count file has >= expected lines matching the regex. A job starts only when every
# job named in "after" is complete. A job counts as running if any python.exe command line contains its args.
# Runs resume from results\raw\<EXP>.done, so a killed job loses only its in-flight tasks.
# The file is re-read every cycle, so stages can be added while the watchdog runs.
# Also logs python.exe processes that appear or disappear (results\watchdog.log) to help find what kills them.
#
# Launch:  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\watchdog.ps1
#          (from a long-lived shell, e.g. a Claude Code background task or a terminal tab; not through WMI)
# Stop:    end the powershell.exe whose command line contains watchdog.ps1

$root = Split-Path -Parent $PSScriptRoot                     # repository root (this script lives in scripts\)
$py   = if ($env:AOD_PYTHON) { $env:AOD_PYTHON } else { (Get-Command python).Source }
$bg   = Join-Path $root 'scripts\bg.py'
$log  = Join-Path $root 'results\watchdog.log'
$jobsFile = Join-Path $root 'results\jobs.txt'
$restarts = @{}
$known = @{}
$maxRestarts = 30

function Say($msg) { Add-Content -Path $log -Value ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + ' ' + $msg) -Encoding UTF8 }

function Count-Done($file, $regex) {
    $p = Join-Path $root $file
    if (-not (Test-Path $p)) { return 0 }
    return @(Select-String -Path $p -Pattern $regex).Count
}

Say "watchdog started (pid $PID)"
$cycle = 0
while ($true) {
    $procs = @(Get-CimInstance Win32_Process -Filter "name='python.exe'")
    # --- process appear/disappear log
    $now = @{}
    foreach ($p in $procs) { $now[[int]$p.ProcessId] = $p.CommandLine }
    foreach ($id in @($known.Keys)) {
        if (-not $now.ContainsKey($id)) {
            $c = $known[$id]; Say ("gone   pid {0}: {1}" -f $id, $c.Substring(0, [Math]::Min(110, $c.Length))); $known.Remove($id)
        }
    }
    foreach ($id in $now.Keys) {
        if (-not $known.ContainsKey($id)) { $known[$id] = [string]$now[$id] }
    }
    # --- job control every 6th cycle (about 30 s)
    if ($cycle % 6 -eq 0 -and (Test-Path $jobsFile)) {
        $jobs = @(Get-Content $jobsFile | Where-Object { $_.Trim() -and -not $_.Trim().StartsWith('#') } | ForEach-Object {
            $f = $_.Split(';') | ForEach-Object { $_.Trim() }
            [pscustomobject]@{ Name = $f[0]; Args = $f[1]; Log = $f[2]; CountFile = $f[3]; Regex = $f[4];
                               Expected = [int]$f[5]; After = @($f[6].Split(',') | Where-Object { $_ }) }
        })
        $complete = @{}
        foreach ($j in $jobs) { $complete[$j.Name] = ((Count-Done $j.CountFile $j.Regex) -ge $j.Expected) }
        foreach ($j in $jobs) {
            if ($complete[$j.Name]) { continue }
            if (@($j.After | Where-Object { -not $complete[$_] }).Count -gt 0) { continue }
            $running = @($procs | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($j.Args) })
            if ($running.Count -gt 0) { continue }
            if (-not $restarts.ContainsKey($j.Name)) { $restarts[$j.Name] = 0 }
            if ($restarts[$j.Name] -ge $maxRestarts) { continue }
            $restarts[$j.Name]++
            # started as our own child: processes started through WMI (Win32_Process.Create) were repeatedly
            # killed together on this machine, while processes in this shell's tree survived
            $r = Start-Process -FilePath $py -ArgumentList (@("`"$bg`"", $j.Log) + $j.Args.Split(' ')) -WorkingDirectory $root `
                               -WindowStyle Hidden -PassThru
            Say ("start  {0} (launch #{1}, pid {2}, done {3}/{4})" -f $j.Name, $restarts[$j.Name], $r.Id,
                 (Count-Done $j.CountFile $j.Regex), $j.Expected)
        }
    }
    $cycle++
    Start-Sleep -Seconds 5
}
