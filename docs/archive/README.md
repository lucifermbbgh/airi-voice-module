# 归档文档（历史快照）

本目录存放**过时的历史快照文档**，内容已被主线文档（各阶段的 DESIGN / DETAILED-DESIGN / TEST-REPORT）覆盖。保留用于追溯历史，不作为当前参考。

| 文档 | 定位 | 归档原因 |
|------|------|---------|
| `SESSION-REPORT-2026-07-24.md` | 单日开发会话快照 | 内容已过时，被各阶段设计/测试文档覆盖 |
| `TASK-PROGRESS-REPORT.md` | Phase 1+2 完成时的进度快照 | 进度已更新到 README「开发阶段」表 |
| `STEP-5-DESIGN.md` | Phase 3 Step 5 中间设计 | 已融入 `PHASE-3-TTS-DESIGN.md` |

> 说明：这类「会话快照 / 进度快照」文档本质上是时间点快照，会随项目推进而过时。
> 当前约定：**会话进度通过 MCP conversation_state 数据库持久化**，文档库只保留
> 「设计（DESIGN）+ 测试（TEST-REPORT）」两类稳定主线文档，避免快照文档堆积。
