# run_experiments.ps1

$levels = @(
    "full-divider_salad",
    "partial-divider_salad",
    "open-divider_salad",
    "full-divider_tomato",
    "partial-divider_tomato",
    "open-divider_tomato",
    "full-divider_tl",
    "partial-divider_tl",
    "open-divider_tl"
)

$models = @("bd", "dc", "fb", "up", "greedy")

$nagents = 2
$nseed = 20

foreach ($seed in 1..$nseed) {
    foreach ($level in $levels) {
        foreach ($model1 in $models) {
            foreach ($model2 in $models) {
                # 打印执行的命令
                Write-Host "python main.py --num-agents $nagents --seed $seed --level $level --model1 $model1 --model2 $model2 --record"
                
                # 执行 Python 命令
                python main.py --num-agents $nagents --seed $seed --level $level --model1 $model1 --model2 $model2  --record
                
                # 等待 5 秒
                Start-Sleep -Seconds 5
            }
        }
    }
}