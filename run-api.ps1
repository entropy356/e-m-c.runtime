#Requires -Version 7.0
<#
.SYNOPSIS
    Demo 3 API 模式启动脚本：魔搭（ModelScope）DeepSeek 做 Chat + Jina 做 Embedding。

.DESCRIPTION
    - 自动从 进程 / 用户 / 系统 环境变量里找 MS_API_KEY 与 JINA_API_KEY，找不到直接报错退出。
    - 设置 AGENT_* 变量；已经存在的变量优先，方便临时覆盖模型或端点。
    - 启动前检查数据库里记录的 embedding profile，避免撞上
      "Embedding provider/model differs from this database; use a new --db path"。

.PARAMETER Database
    数据库路径，默认 data/agent.sqlite3（相对本脚本所在目录）。

.PARAMETER Conversation
    会话 ID，默认 default。不同 ID 的消息历史隔离，长期记忆共享。

.PARAMETER ChatModel
    魔搭 API-Inference 的模型 ID，默认 deepseek-ai/DeepSeek-V4-Flash-Vision-Exp。

.PARAMETER EmbedModel
    Jina 的 embedding 模型，默认 jina-embeddings-v3（1024 维）。

.PARAMETER Protocol
    认知协议，默认 tool。tool = 调用 submit_cognition 工具；json = 文本 JSON + response_format；
    text = 纯文本 JSON。不做自动降级，网关不支持时直接报错，请显式换一个协议。

.PARAMETER Events
    给 chat 加 --events：开一个本地 loopback 事件端口，并把端口写进 <Database>.events.json。
    另一个窗口用 watch 连上去，每次提交后由主进程推送刷新。

.PARAMETER Port
    watch 连接的事件端口；省略时自动读取 <Database>.events.json。

.PARAMETER Rounds
    watch 显示的最近轮次数，默认 6。

.PARAMETER NoClear
    watch 不清屏，便于把每次刷新追加到日志。

.EXAMPLE
    .\run-api.ps1
    进入交互式聊天（等价于 chat）。

.EXAMPLE
    .\run-api.ps1 turn "我更喜欢简短的书面进度，每周五发一次。"

.EXAMPLE
    .\run-api.ps1 history -Database data/other.sqlite3 -Conversation project-x

.EXAMPLE
    .\run-api.ps1 chat -Protocol json
    网关不支持 tools 时改用文本 JSON 协议。

.EXAMPLE
    .\run-api.ps1 memories -Database data/agent.sqlite3
    只读查看长期记忆（不需要 Key，也不校验 embedding profile）。

.EXAMPLE
    .\run-api.ps1 memories -Database data/agent.sqlite3 -MemoryId 2 -Decay
    查看某条记忆的完整字段，并在列表里显示衰减预测列。

.EXAMPLE
    .\run-api.ps1 context -Database data/agent.sqlite3
    查看该会话的消息顺序与每轮记忆活动。

.EXAMPLE
    .\run-api.ps1 context -Database data/agent.sqlite3 -Payload
    查看下一轮模型实际收到的 system 提示词、记忆块与历史消息。

.EXAMPLE
    .\run-api.ps1 chat -Events
    监控模式的主进程：开事件端口，并打印 watch 的连接命令。

.EXAMPLE
    .\run-api.ps1 watch -Database data/agent.sqlite3
    另一个窗口：阻塞等待推送，每次提交后刷新一次（不轮询数据库）。

.EXAMPLE
    .\run-api.ps1 watch -Json -NoClear > watch.jsonl
    把每次刷新记录成一行 JSON，便于事后分析。

.EXAMPLE
    .\run-api.ps1 watch -Payload
    监控切到完整上下文：system 提示词 + 记忆块 + 全部历史，按实际发送顺序排列。

.NOTES
    交互式聊天直接运行即可。若要把文件内容批量喂给 chat，请绕过本脚本：
    Get-Content prompts.txt | python -m long_term_memory --mode api --db data/agent.sqlite3 chat
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Database = 'data/agent.sqlite3',
    [string]$Conversation = 'default',
    [string]$ChatModel = 'deepseek-ai/DeepSeek-V4-Flash-Vision-Exp',
    [string]$EmbedModel = 'jina-embeddings-v3',
    [string]$ChatBase = 'https://api-inference.modelscope.cn/v1',
    [string]$EmbedBase = 'https://api.jina.ai/v1',
    [ValidateSet('tool', 'json', 'text')]
    [string]$Protocol = $null,
    [int]$MemoryId = 0,
    [ValidateSet('weight', 'id', 'age', 'activations')]
    [string]$Sort = '',
    [switch]$Decay,
    [switch]$Full,
    [switch]$Json,
    [switch]$Payload,
    [switch]$Events,
    [int]$Port = 0,
    [int]$Rounds = 0,
    [switch]$NoClear,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

$ErrorActionPreference = 'Stop'

# 控制台按 UTF-8 输出，避免中文变成乱码
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath $PSScriptRoot

function Get-EnvValue {
    param([string]$Name)
    foreach ($scope in 'Process', 'User', 'Machine') {
        $value = [Environment]::GetEnvironmentVariable($Name, $scope)
        if ($value) { return $value }
    }
    return $null
}

function Fail {
    param([string]$Message)
    Write-Host "[错误] $Message" -ForegroundColor Red
    exit 1
}

if (-not $Command -or $Command.Count -eq 0) { $Command = @('chat') }

# memories / context / watch 是纯只读查看，不需要 Key，也不需要校验 embedding profile
$viewOnly = $Command[0] -in @('memories', 'context', 'watch', 'monitor')

if (-not $viewOnly) {
    $msKey = Get-EnvValue 'MS_API_KEY'
    $jinaKey = Get-EnvValue 'JINA_API_KEY'

    if (-not $msKey) {
        Fail "找不到 MS_API_KEY。请先设置（用户级）：[Environment]::SetEnvironmentVariable('MS_API_KEY','ms-xxx','User')"
    }
    if (-not $jinaKey) {
        Fail "找不到 JINA_API_KEY。请先设置（用户级）：[Environment]::SetEnvironmentVariable('JINA_API_KEY','jina_xxx','User')"
    }

    # 已有值优先，未设置才用脚本默认值
    if (-not $env:AGENT_CHAT_BASE) { $env:AGENT_CHAT_BASE = $ChatBase }
    if (-not $env:AGENT_CHAT_MODEL) { $env:AGENT_CHAT_MODEL = $ChatModel }
    if (-not $env:AGENT_CHAT_KEY) { $env:AGENT_CHAT_KEY = $msKey }
    if (-not $env:AGENT_EMBED_PROTOCOL) { $env:AGENT_EMBED_PROTOCOL = 'openai' }
    if (-not $env:AGENT_EMBED_BASE) { $env:AGENT_EMBED_BASE = $EmbedBase }
    if (-not $env:AGENT_EMBED_MODEL) { $env:AGENT_EMBED_MODEL = $EmbedModel }
    if (-not $env:AGENT_EMBED_KEY) { $env:AGENT_EMBED_KEY = $jinaKey }
    # 显式传了 -Protocol 就以参数为准；否则沿用已有环境变量，最后兜底 tool
    if ($Protocol) { $env:AGENT_COGNITION_PROTOCOL = $Protocol }
    elseif (-not $env:AGENT_COGNITION_PROTOCOL) { $env:AGENT_COGNITION_PROTOCOL = 'tool' }

    # 数据库里写死了 embedding profile，换模型/维度必须换库
    $expectedProfile = "api:$($env:AGENT_EMBED_PROTOCOL):$($env:AGENT_EMBED_MODEL):native"
    if (Test-Path -LiteralPath $Database) {
        $stored = & python -c "import sqlite3,sys;print(sqlite3.connect(sys.argv[1]).execute('select embedding_profile from runtime where id=1').fetchone()[0])" $Database 2>$null
        if ($LASTEXITCODE -eq 0 -and $stored) {
            $stored = $stored.Trim()
            if ($stored -ne $expectedProfile) {
                Write-Host "[错误] 数据库 $Database 绑定的 embedding profile 是：" -ForegroundColor Red
                Write-Host "        $stored" -ForegroundColor Red
                Write-Host "        当前配置是：$expectedProfile" -ForegroundColor Red
                Write-Host "        请换一个 -Database，或删掉该库重新开始。" -ForegroundColor Yellow
                exit 1
            }
        }
    }

    Write-Host ("chat  = " + $env:AGENT_CHAT_MODEL) -ForegroundColor DarkGray
    Write-Host ("embed = " + $env:AGENT_EMBED_MODEL) -ForegroundColor DarkGray
    Write-Host ("proto = " + $env:AGENT_COGNITION_PROTOCOL + "  (tool | json | text，不做自动降级)") -ForegroundColor DarkGray
    Write-Host ("db    = " + $Database + "  (conversation: " + $Conversation + ")") -ForegroundColor DarkGray
}
else {
    Write-Host ("db    = " + $Database + "  (只读查看)") -ForegroundColor DarkGray
    $viewArgs = @('-m', 'long_term_memory', '--db', $Database)
    # 全局参数（--conversation / --protocol）必须写在子命令之前
    if ($Command[0] -in @('context', 'watch', 'monitor')) {
        $viewArgs += @('--conversation', $Conversation)
        if ($Protocol) { $viewArgs += @('--protocol', $Protocol) }
    }
    if ($Command[0] -eq 'context') {
        $viewArgs += 'context'
        if ($Payload) { $viewArgs += '--payload' }
    }
    elseif ($Command[0] -in @('watch', 'monitor')) {
        $viewArgs += 'watch'
        if ($Port -gt 0) { $viewArgs += @('--port', [string]$Port) }
        if ($Sort) { $viewArgs += @('--sort', $Sort) }
        if ($Rounds -gt 0) { $viewArgs += @('--rounds', [string]$Rounds) }
        if ($NoClear) { $viewArgs += '--no-clear' }
        if ($Payload) { $viewArgs += '--payload' }
        if ($Decay) { $viewArgs += '--decay' }
    }
    else {
        $viewArgs += 'memories'
        if ($MemoryId -gt 0) { $viewArgs += [string]$MemoryId }
        if ($Sort) { $viewArgs += @('--sort', $Sort) }
        if ($PSBoundParameters.ContainsKey('Conversation')) { $viewArgs += @('--conversation', $Conversation) }
        if ($Decay) { $viewArgs += '--decay' }
    }
    if ($Full) { $viewArgs += '--full' }
    if ($Json) { $viewArgs += '--json' }
    & python @viewArgs
    exit $LASTEXITCODE
}

$pythonArgs = @('-m', 'long_term_memory', '--mode', 'api', '--db', $Database, '--conversation', $Conversation) + $Command
if ($Events -and $Command[0] -eq 'chat') { $pythonArgs += '--events' }
& python @pythonArgs
exit $LASTEXITCODE
