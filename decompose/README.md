# Text-to-SQL CoT 忠实性评估与纠正实验

## 实验目标

检测并修复 Text-to-SQL 思维链中的"虚假正确"（Spurious Correctness）问题——即最终 SQL 执行结果正确，但中间推理链存在逻辑错误的情况。

## 流水线总览

```
decomposed_cot_final.json (5361 条样本，含分解后的 CoT 步骤)
         │
         ▼
  evaluate_decomposed_cot.py  ─── PRM 逐步骤评估
         │
         ▼
  step_level_evaluation_results_*.json  (每条步骤的 is_valid + flaw_type)
         │
         ▼
  select_error_cot.py  ─── 筛选 overall_valid=false 的样本
         │
         ▼
  error_samples_only_*.json  (仅含错误样本)
         │
         ▼
  correct.py  ─── LLM 纠正错误步骤 → 重评估
         │
         ▼
  self_correction_results_*.json + self_correction_report_*.json
```

## 文件说明

### 核心脚本

| 文件 | 作用 |
| --- | --- |
| [evaluate_decomposed_cot.py](evaluate_decomposed_cot.py) | **PRM 评估器**。逐步骤调用 DeepSeek API，判断每一步是否存在 Schema_Hallucination / Evidence_Violation / Logic_Error / CoT_SQL_Disconnect。10 线程并发。 |
| [correct.py](correct.py) | **纠正引擎**。对错误步骤调用 LLM 修正，然后重新跑 PRM 评估整条链。支持 Blind Mode（不提供 final_sql，防逆拟合）。 |
| [select_error_cot.py](select_error_cot.py) | **筛选器**。从全量评估结果中筛出 overall_valid=false 的样本，供纠正阶段使用。 |
| [decompose_cot.py](decompose_cot.py) | **CoT 分解器**。将原始完整思维链拆分为 numbered steps。 |
| [enrich_decomposed_data.py](enrich_decomposed_data.py) | 为分解后的数据补充 schema、question、evidence 等元信息。 |
| [sample_for_human_review.py](sample_for_human_review.py) | 人工抽检抽样脚本。分三类（PRM 通过/纠正成功/纠正失败）各抽 15 条。 |

### 数据文件（按流水线顺序）

| 文件 | 说明 |
| --- | --- |
| `decomposed_cot.json` | CoT 分解中间产物 |
| `decomposed_cot_final.json` | **主数据集**。5361 条样本，含 decomposed_steps、schema、generated_sql 等。流水线起点。 |
| `step_level_evaluation_results.json` | 旧 PRM 提示词（v1，已归档）的全量评估结果 |
| `step_level_evaluation_results_1.json` | **当前 PRM 提示词**（紧凑版）的全量评估结果，位于 `simplify_prompt_result/` |
| `error_samples_only.json` | 旧 PRM 筛出的 112 条错误样本 |
| `error_samples_only_1.json` | 新 PRM 筛出的 78 条错误样本，位于 `simplify_prompt_result/` |
| `self_correction_results.json` | Non-blind 纠正结果（旧 prompt） |
| `self_correction_results_blind.json` | Blind 纠正结果（旧 prompt，112 条） |
| `self_correction_report.json` | 对应的统计报告 |
| `self_correction_report_blind.json` | 对应的统计报告 |
| `human_review_samples.json` | 人工抽检的 45 条样本（A/B/C 三类各 15） |

### 子目录

| 目录 | 内容 |
| --- | --- |
| `simplify_prompt_result/` | 使用紧凑版 PRM 提示词重跑的全部结果（评估+筛选+纠正） |
| `out/` | 最近一次运行结果（旧 PRM 筛选 → 新提示词纠正，112 条） |

### 辅助文件

| 文件 | 说明 |
| --- | --- |
| `实验结果.md` | 早期实验记录（112 条，旧 prompt） |
| [prompts_v1.md](prompts_v1.md) | 原始长版 PRM system prompt 存档（约 60 行） |

## 四种错误类型

| 类型 | 含义 | 示例 |
| --- | --- | --- |
| Schema_Hallucination | 引用了 schema 中不存在的表名或列名 | "join on EmployeeID"（实际列名是 BusinessEntityID） |
| Evidence_Violation | 无视或推翻 External Knowledge 中的业务规则 | Evidence 说 class_label='M'，步骤用了 'ML' |
| Logic_Error | 逻辑推导错误、无效 JOIN、仓促下"不可能"结论 | 声称"没有表可以 JOIN"但实际有 |
| CoT_SQL_Disconnect | 步骤描述的 SQL 操作与 Final SQL 矛盾或遗漏关键操作 | 步骤说 LEFT JOIN，SQL 却是 INNER JOIN |

## 关键实验参数

- **模型**: DeepSeek API (`deepseek-chat`), temperature=0.0, response_format=json_object
- **并发**: 10 workers (ThreadPoolExecutor)
- **重试**: 最多 3 次
- **Blind Mode**: 纠正时不提供 final_sql，避免模型逆拟合

## 关键发现

1. **Blind vs Non-Blind 修复率差距小** (70.54% vs 72.32%)，说明原始非盲结果未显著高估。
2. **Evidence_Violation 修复率最低**（53-72%），常因 Evidence 本身与 SQL/question 冲突导致。
3. **紧凑版 PRM 提示词**（决策表替代 5 条冗余规则）修复率提升约 3.5 个百分点。
4. **纠正的级联问题**：只修正被标记的错误步骤，后续步骤中对被修正内容的引用不会自动更新。
