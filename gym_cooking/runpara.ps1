 # run_experiments_405_parallel_with_runtime.ps1

# 1. 配置参数
$levels   = @(
    "full-divider_salad","partial-divider_salad","open-divider_salad",
    "full-divider_tomato","partial-divider_tomato","open-divider_tomato",
    "full-divider_tl","partial-divider_tl","open-divider_tl"
)
 # $models   = @("bd","dc","fb","up","greedy")
$models   = @("bd")
$nagents  = 2
$nseed    = 9    # seeds 1..9

# 2. 准备 runtime 日志文件
$runtimeFile = Join-Path $PSScriptRoot 'runtime.txt'
if (-not (Test-Path $runtimeFile)) {
    "Index,Seed,Level,Model,Command,DurationSeconds" |
        Out-File -FilePath $runtimeFile -Encoding utf8
}

# 3. 构造任务列表（共 9 × 9 × 5 = 405 条）
$totalTasks = $nseed * $levels.Count * $models.Count
$tasks = New-Object System.Collections.ObjectModel.ObservableCollection[PSCustomObject]

[int]$idx = 0
foreach ($seed in 1..$nseed) {
    foreach ($level in $levels) {
        foreach ($model in $models) {
            $idx++
            $tasks.Add([PSCustomObject]@{
                Index = $idx
                Seed  = $seed
                Level = $level
                Model = $model
            })
        }
    }
}

Write-Host "[INFO] Total tasks to run: $totalTasks"

# 4. 并行执行，记录命令和耗时
$tasks |
  ForEach-Object -Parallel {
    # 构造命令：model1=model2
    $cmd = "python main.py --num-agents $using:nagents --seed $($_.Seed) " +
           "--level $($_.Level) --model1 $($_.Model) --model2 $($_.Model) --record   --max-num-timesteps 45"

    # 开始计时
    $startTime = Get-Date

    # 执行命令
    Invoke-Expression $cmd

    # 正确计算持续秒数
    $endTime  = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds

    # 控制台日志
    Write-Host ("[DONE ] Task {0}/{1} finished at {2:HH:mm:ss}, duration: {3:N1}s" -f `
        $_.Index, $using:totalTasks, $endTime, $duration)

    # 追加到 runtime.txt
    $escapedCmd = $cmd.Replace('"', '""')
    $line = "{0},{1},{2},{3},""{4}"",{5:N1}" -f `
        $_.Index, $_.Seed, $_.Level, $_.Model, $escapedCmd, $duration
    Add-Content -Path $using:runtimeFile -Value $line
  } -ThrottleLimit 8

# 5. 全部结束
Write-Host "[INFO] All $totalTasks experiments done."


 
