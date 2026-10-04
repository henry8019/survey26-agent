# 巡天智能体

基于 GOSIM 官方 Python 示例 `examples-2026-10-02`；来源和下载哈希见 `UPSTREAM.json`。
使用 Python 3.12 标准库，无第三方运行依赖。启动方式仍为 `python3 -u agent.py`，
通过 JSONL-v4 逐轮接收公开输入。官方规则、当前卡配置及评分器决定行为，示例作为参考。
当前修复、真实模型回归及官方输入核对见 `KNOWN_ISSUES.md`；历史实验见
`SCHEDULING.md`、`OPTIMIZATION.md`。

## 策略与模型

程序负责光纤几何、线性边际收益、曝光时长及合法动作。历史最高得分与完成因子范围独立维护，
只有完成因子下界达到当前卡门槛才确认必观测完成。反馈先入账，再按更正撤销曝光并重建状态。

规划器对每格多个目标联合搜索曝光，补充科学收益率、必观测成本和密度锚点。
沿用整体通过评测的质量校准版本，包括少量第二步前瞻；不能将组合收益归因于前瞻单项。
独立移除前瞻的组合未通过复跑，详情见 KNOWN_ISSUES.md。替代方案保留首选方案预计可达标的必观测目标；
质量异常或强制诊断期间保留原选点方式。试算不写入反馈状态，只有最终动作提交预测。
故障证据按曝光和观测夜汇总，报告得失读取当前配置；局部零分区域具有失效时间。
可能饱和的反馈不作为精确效率样本；缺少未饱和反馈时，受限短曝光补充证据，
保留原动作可完成的必观测目标。最后一个观测夜开始时，在报告不会受罚且运行预算允许时，
检查遗留故障；report 不消耗模拟观测时间，误报后停止该检查。

Kimi 负责两个实际环节：解释有来源和有效期的公告、预报、请求；在新请求、质量下降、
连续未命中或数据更正后调整规划优先级。模型不能直接发动作或修改已确认完成状态。
公告建议随当前公告更新；预报建议只适用于列出的观测夜。

每卡模型预算 150 秒、最多 40 次 HTTP 尝试；单次超时 8 秒，每个问题最多两次尝试，
最后 60 秒留给程序运行。消息理解最多 24 次，计划调整最多 16 次。
调用失败、结构非法、过期或预算用完时，使用程序策略。

## 本地配置与验证

仅在没有 `.env` 时执行 `Copy-Item .env.example .env`，随后编辑本地文件填写
`OPENAI_API_KEY`。接口 `https://api.kimi.com/coding/v1`，模型 `k3`；不要将密钥写入代码。
平台使用网站配置并注入的环境变量。K3 请求使用 `reasoning_effort=none`，不设置温度。

```powershell
py -3.12 tools/prepare_local.py
py -3.12 tools/prepare_official.py
py -3.12 -m unittest discover -s tools -p test_*.py
py -3.12 .local/runner/verify_engine.py
# 默认官方 alpha-delta；当前官网仅提供六个文件，天气不公开，本地完整评分会提前停止
py -3.12 tools/freeze_version.py .local/versions/my-candidate
py -3.12 tools/evaluate.py --project .local/versions/my-candidate --out run_output/my-evaluation
# 示例 L1-L4 只用于工程回归，需显式指定
py -3.12 tools/evaluate.py --card-set examples --project .local/versions/my-candidate --out run_output/example-evaluation
```

新克隆的仓库需先执行 `tools/prepare_local.py`。它下载 `UPSTREAM.json` 指定的官方资源，
校验完整 ZIP 的 SHA-256 后，仅把模拟器和本地卡放入被忽略的 `.local/`。
已有文件如与官方版本不同会停止，保留原文件。比赛容器运行 `agent.py` 不需要这些资源。

结果保留各卡分项得分、漏观数、请求奖励、源码哈希、耗时及无密钥的模型输入输出。
`tools/run_practice.py --card L1 --mock-model` 仅验证协议；模拟模型分数不能替代真实 Kimi 对照。
Windows 管道适配位于 `tools/windows_transport.py`，原版评分引擎未修改。

随示例下载的 L1–L4 是本地卡，四卡都为 16 根光纤，不能视作平台练习卡 α–δ、线上卡 A–D
或隐藏卡 E–H。其他光纤数、间隙、站点和参数以合成输入测试，尚无对应真实卡实测成绩。
智能体不读取本地天气答案，不按卡片或目标 ID 定制策略。

## 限时请求与实验边界

请求进度只计发布后、截止前的有效曝光，并在数据更正后重建。规划器比较完整请求组合的
奖励、可行时间、观测夜内等待及科学收益机会成本；按实际动作重检，不能破坏必观测目标。
缺少模型密钥时仍可按程序策略输出合法动作，但正式评测需配置密钥才能运行两个模型环节。

第三轮的窗口优先、指向校准和增强诊断未通过真实评测验收，保留在 `experiments/stage3/`，
不进入提交包。运行路径未引用的 `agent_core/calibration.py` 保留为与评测版本一致的辅助文件。

## 提交包

```powershell
py -3.12 pack_agent.py --out ../survey-agent-known-fixes.zip
```

ZIP 根目录包含 `observer.project.json`；排除 `.env`、`.local/`、`experiments/`、测试工具和运行结果。
仅将通过验收的版本打包；原 survey-agent.zip 保留为 v2 回退包。
平台上传和最终版本选择另行处理。
官方原说明保留在 `README.zh.md`，署名及许可见 `LICENSE.md`。
