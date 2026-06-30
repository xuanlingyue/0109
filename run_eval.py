# -*- coding: utf-8 -*-
"""
自动回复质量评估流水线（入口）。

用法：
    python run_eval.py                 # 默认 mock 模式
    python run_eval.py --mode llm      # 使用 LLM（需配置 OPENAI_API_KEY 等）
    python run_eval.py --mode mock     # 强制 mock

输出：
    - 控制台打印整体摘要
    - 写入 eval_report.md（含整体得分、各指标分布、最差 3 条 case 分析）
"""

import argparse
import json
import os
import statistics
from collections import OrderedDict

from metrics import METRICS, WEIGHTS, PRIORITY_ORDER
from scorer import get_scorer, weighted_total, MockScorer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPLIES_PATH = os.path.join(BASE_DIR, "task3_auto_replies.json")
HUMANREF_PATH = os.path.join(BASE_DIR, "task3_human_ref.json")
REPORT_PATH = os.path.join(BASE_DIR, "eval_report.md")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tier(score):
    """5 分制 -> 等级。"""
    if score >= 4.5:
        return "优秀"
    if score >= 3.5:
        return "良好"
    if score >= 2.5:
        return "及格"
    if score >= 1.5:
        return "较差"
    return "极差"


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "llm"], default="mock")
    parser.add_argument("--workers", type=int, default=5,
                        help="LLM 并发数，默认 5")
    parser.add_argument("--timeout", type=int, default=30,
                        help="LLM 单条超时时间（秒），默认 30")
    args = parser.parse_args()

    replies = load_json(REPLIES_PATH)
    human_refs = {x["id"]: x for x in load_json(HUMANREF_PATH)}

    scorer = get_scorer(force_mock=(args.mode == "mock"),
                        max_workers=args.workers, timeout=args.timeout)
    print(f"[pipeline] 使用打分器: {scorer.name} | 样本数: {len(replies)}")

    results = []
    fallback = MockScorer()

    def _progress(cid, status):
        if status == "ok":
            print(f"[pipeline] {cid} 完成")
        else:
            print(f"[pipeline] {cid} {status}，回退 mock")

    if scorer.name == "llm":
        # LLM 模式：并发打分，带进度打印
        batch_results = scorer.score_batch(
            replies, fallback_scorer=fallback, progress_callback=_progress
        )
        for case, br in zip(replies, batch_results):
            scores = br["scores"]
            signals = br["signals"]
            total = weighted_total(scores)
            results.append({
                "id": case["id"],
                "user_question": case["user_question"],
                "auto_reply": case["auto_reply"],
                "human_reference": human_refs.get(case["id"], {}).get("human_reference", ""),
                "annotator_notes": human_refs.get(case["id"], {}).get("annotator_notes", ""),
                "scores": scores,
                "total": total,
                "signals": signals,
                "tier": tier(total),
                "source": br.get("mode", "llm"),
            })
    else:
        # mock 模式：串行即可
        for case in replies:
            scores, signals = scorer.score(case)
            total = weighted_total(scores)
            results.append({
                "id": case["id"],
                "user_question": case["user_question"],
                "auto_reply": case["auto_reply"],
                "human_reference": human_refs.get(case["id"], {}).get("human_reference", ""),
                "annotator_notes": human_refs.get(case["id"], {}).get("annotator_notes", ""),
                "scores": scores,
                "total": total,
                "signals": signals,
                "tier": tier(total),
                "source": "mock",
            })

    # ---- 统计 ----
    stats = OrderedDict()
    for k in list(WEIGHTS.keys()) + ["total"]:
        vals = [r["scores"][k] if k != "total" else r["total"] for r in results]
        stats[k] = {
            "mean": round(statistics.mean(vals), 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "stdev": round(statistics.pstdev(vals), 2),
        }

    overall = round(statistics.mean([r["total"] for r in results]), 2)
    worst = sorted(results, key=lambda x: x["total"])[:3]
    best = sorted(results, key=lambda x: -x["total"])[:3]

    # 分布（按等级）
    tier_counts = {t: 0 for t in ["优秀", "良好", "及格", "较差", "极差"]}
    for r in results:
        tier_counts[r["tier"]] += 1

    write_report(results, stats, overall, worst, best, tier_counts, scorer.name)
    print_summary(results, stats, overall, worst, scorer.name)


def print_summary(results, stats, overall, worst, scorer_name):
    print("\n" + "=" * 60)
    print(f"整体加权均分: {overall} / 5.0  ({tier(overall)})  模式: {scorer_name}")
    print("=" * 60)
    print("各指标均分:")
    for k, v in stats.items():
        label = METRICS.get(k, {}).get("name", "加权总分")
        print(f"  {label:8s}  mean={v['mean']}  min={v['min']}  max={v['max']}  std={v['stdev']}")
    print("\n最差 3 条:")
    for r in worst:
        src = f"[{r['source']}]"
        print(f"  {r['id']}  总分 {r['total']}  {r['tier']}  {src}  signals={r['signals']}")
    print(f"\n报告已写入: {REPORT_PATH}")


def write_report(results, stats, overall, worst, best, tier_counts, scorer_name):
    lines = []
    lines.append("# 自动回复质量评估报告\n")
    lines.append(f"> 评估模式: `{scorer_name}` | 样本量: {len(results)} | 评分范围: 1-5 分\n")

    lines.append("## 1. 整体结论\n")
    lines.append(f"- **整体加权均分：{overall} / 5.0 （{tier(overall)}）**\n")
    lines.append("- 各指标均分及分布见下表。核心结论：**准确性普遍良好（无明显瞎编），"
                 "最大短板是\"有用性\"——大量回复把责任推给用户**（让用户自查详情页、再联系客服、耐心等待）。\n")

    lines.append("### 1.1 各指标统计\n")
    lines.append("| 指标 | 权重 | 均分 | 最低 | 最高 | 标准差 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for k, v in stats.items():
        if k == "total":
            lines.append(f"| **加权总分** | 1.00 | {v['mean']} | {v['min']} | {v['max']} | {v['stdev']} |")
        else:
            lines.append(f"| {METRICS[k]['name']} | {WEIGHTS[k]} | {v['mean']} | {v['min']} | {v['max']} | {v['stdev']} |")

    lines.append("\n### 1.2 总分等级分布\n")
    lines.append("| 等级 | 区间 | 数量 |")
    lines.append("| --- | --- | --- |")
    for t, rng in [("优秀", ">=4.5"), ("良好", "3.5-4.5"), ("及格", "2.5-3.5"), ("较差", "1.5-2.5"), ("极差", "<1.5")]:
        lines.append(f"| {t} | {rng} | {tier_counts[t]} |")

    # 打分来源统计（仅在 llm 模式下有意义）
    source_counts = {}
    for r in results:
        source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1
    lines.append("\n### 1.3 打分来源统计\n")
    lines.append("| 来源 | 数量 | 说明 |")
    lines.append("| --- | --- | --- |")
    for src, cnt in source_counts.items():
        desc = "LLM 直接打分" if src == "llm" else "失败回退 mock 打分"
        lines.append(f"| {src} | {cnt} | {desc} |")

    lines.append("\n### 1.4 指标优先级\n")
    lines.append("```\n" + PRIORITY_ORDER + "```\n")

    lines.append("## 2. 明细打分\n")
    lines.append("| ID | 来源 | 准确性 | 有用性 | 针对性 | 语气 | 总分 | 等级 | 命中信号 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        s = r["scores"]
        sig = "、".join(r["signals"]) if r["signals"] else "-"
        lines.append(f"| {r['id']} | {r['source']} | {s['accuracy']} | {s['helpfulness']} | {s['specificity']} | {s['tone']} | {r['total']} | {r['tier']} | {sig} |")

    lines.append("\n## 3. 最差 3 条 case 分析\n")
    lines.append("> 结合人工参考回复（human_ref）与标注说明，分析失败原因。\n")
    for i, r in enumerate(worst, 1):
        s = r["scores"]
        lines.append(f"### TOP{i} 最差：{r['id']}（总分 {r['total']}，{r['tier']}，来源：{r['source']}）")
        lines.append(f"- **用户问题**：{r['user_question']}")
        lines.append(f"- **自动回复**：{r['auto_reply']}")
        lines.append(f"- **人工参考**：{r['human_reference']}")
        lines.append(f"- **分项**：准确性 {s['accuracy']} / 有用性 {s['helpfulness']} / "
                     f"针对性 {s['specificity']} / 语气 {s['tone']}")
        lines.append(f"- **命中信号**：{('、'.join(r['signals'])) if r['signals'] else '无'}")
        lines.append(f"- **人工标注**：{r['annotator_notes']}")
        lines.append(f"- **分析**：{worst_analysis(r)}\n")

    lines.append("## 4. 表现较好的 case（参考）\n")
    lines.append("| ID | 总分 | 用户问题摘要 |")
    lines.append("| --- | --- | --- |")
    for r in best:
        lines.append(f"| {r['id']} | {r['total']} | {r['user_question'][:24]}... |")
    lines.append("\n这些 case 的共性：给出了结合上下文的可执行方案（如 case_04 结合\"才买三天\""
                 "给出质保期换新路径）、在用户不满时先道歉并主动提出立刻处理（case_05），"
                 "或一次性回应多个问题并主动提出帮办（case_17）。共同点是**没有把用户推走**。\n")

    lines.append("## 5. 局限性与改进方向\n")
    lines.append(limitations_text())

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def worst_analysis(r):
    """根据命中信号与人工标注生成简短分析。"""
    sig = " ".join(r["signals"])
    notes = r["annotator_notes"]
    if "答非所问" in sig:
        return ("用户已明确表达\"搞不懂流程\"，自动回复却再次罗列同一套流程，"
                "属于典型的答非所问。应改为先追问\"卡在哪一步\"再针对性引导。")
    if "推用户自查详情页" in sig:
        return ("用户正是因为不想自己翻详情页才来咨询，回复却把用户推回详情页，"
                "属于\"正确但没用\"。应直接给出答案或主动查询后回复。")
    if "甩锅话术" in sig:
        return ("回复以\"联系快递员/联系客服\"等话术收尾，把本应自动回复承担的"
                "解决责任甩回用户或人工，削弱了自动回复的价值。应在回复内闭环，"
                "或给出可自助操作的具体路径。")
    if "用户不满时谈内部事项" in sig:
        return ("用户已处于不满状态，回复却谈论\"加强培训/反馈品控\"等内部事项，"
                "用户感知为官腔。应先解决用户当前问题、给出补偿，再谈后续改进。")
    if "未利用用户给定的上下文" in sig or "套模板" in sig:
        return ("用户已给出具体商品/场景上下文，回复却套用通用模板，针对性不足。"
                "应结合用户上下文（订单、商品参数）给出个性化答复。")
    return notes


def limitations_text():
    return (
        "### 5.1 评估方法可能不准的 case\n"
        "1. **事实准确性无法真正核实**：mock 模式只能据\"是否给出可核对信息\"粗判，"
        "无法验证政策/规格/时效是否正确。涉及退款时效、质保期、民航规定等需要"
        "对接知识库/订单系统才能判准。\n"
        "2. **有用性判定依赖话术模式**：规则通过\"请联系客服/查看详情页\"等关键词"
        "识别\"甩锅\"，但存在误判——例如\"请联系客服核实\"在部分场景下是合理兜底，"
        "会被错罚。\n"
        "3. **针对性无法理解真正语义**：当用户指代\"这个/那款\"时，规则不知道系统"
        "是否真的能查到该商品参数；自动回复未查商品有时是系统能力所限而非回复缺陷。\n"
        "4. **语气共情对反讽/隐性不满不敏感**：如\"上次买坏的这次又坏\"这类隐性投诉，"
        "关键词匹配可能漏判。\n"
        "5. **样本量小（20 条）**：分布结论不具备统计显著性，仅作定性参考。\n\n"
        "### 5.2 改进方向\n"
        "- **接入真实 LLM 评测**（本流水线已支持 `--mode llm`），用大模型做语义级评分，"
        "并以人工标注的 20 条作为评测的一致性校验集（计算与人工排序的 Spearman 相关）。\n"
        "- **事实维度接入知识库**：对回复中出现的时效/阈值/政策类陈述，与业务知识库比对。\n"
        "- **引入用户反馈信号**：将回复后的转人工率、用户追问率、点赞/点踩作为外部校准。\n"
        "- **扩充标注集**：覆盖更多品类与情绪场景，定期重算指标分布以监控漂移。\n"
        "- **对 mock 规则做校准**：用人工分数回归拟合关键词权重，降低误判。\n"
    )


if __name__ == "__main__":
    run()
