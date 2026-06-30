# -*- coding: utf-8 -*-
"""
打分器实现：
  - MockScorer: 基于规则的打分器，零依赖、确定性，默认使用。
  - LLMScorer:  调用 OpenAI 兼容接口（如 DeepSeek / OpenAI）打分，可选。
  - get_scorer: 工厂，根据环境变量自动选择。

设计要点：
  人工标注（task3_human_ref.json）一致地暴露出自动回复的核心短板——
  "把责任推给用户"（请联系客服 / 查看商品详情页 / 建议耐心等待）以及
  "忽略用户已给出的具体上下文直接套模板"。MockScorer 的规则即围绕这两类
  信号设计，并对"用户已表达困难却重复流程"这类答非所问给予重罚。
"""

import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from metrics import METRICS, WEIGHTS


# 加载 .env 文件（不依赖 python-dotenv）
def _load_env_file(path=".env"):
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_file()


# --------------------------------------------------------------------------- #
# 规则信号
# --------------------------------------------------------------------------- #
# "甩锅"类话术：把本应客服完成的事推给用户
ESCAPE_HATCH_PATTERNS = [
    "请联系客服", "联系我们的客服", "联系客服核实", "联系客服获取",
    "联系客服协助", "联系客服询问", "如有疑问请联系客服", "如有疑问，请联系客服",
    "如有其他问题请联系客服", "如有需要，可以联系客服", "如有疑问请联系",
    "联系快递员", "品牌官方客服",
]

# "推去自查"类话术：让用户自己翻商品详情页找答案（注意：订单详情页查进度
# 属合理的自助服务，不在此列，不应被罚）
PUSH_TO_PAGE_PATTERNS = [
    "查看商品详情页", "商品详情页的参数", "商品详情页查看",
    "商品详情页的用户评价", "商品详情页的参数说明",
]

PATIENCE_PATTERNS = ["建议您耐心等待", "请您耐心等待"]

# 用户愤怒/受挫信号
USER_FRUSTRATION_PATTERNS = [
    "态度太差", "没人理我", "搞半天", "太复杂了", "又是坏的", "连续",
    "等了20分钟", "等了", "都不理",
]

# 用户在谈内部事项而非先解决问题（在用户不满场景下属负向）
INTERNAL_TALK_PATTERNS = ["加强客服团队", "加强培训", "反馈给品控部门", "转达给产品团队"]

PROACTIVE_PATTERNS = [
    "我帮您", "我现在就帮您", "我直接帮您", "可以帮您",
    "帮您联系", "帮您查询", "帮您操作", "帮您推荐", "帮您对比", "帮您确认",
]


def _count_any(text, patterns):
    """统计 patterns 命中数。注意：直接 sum 会因子串重叠重复计数
    （如"请联系客服"与"如有疑问请联系客服"），故改用正则按长度降序匹配，
    保证同一处文本只计一次。"""
    pats = sorted(set(patterns), key=len, reverse=True)
    regex = re.compile("|".join(re.escape(p) for p in pats))
    return len(regex.findall(text))


def _has(text, patterns):
    return any(p in text for p in patterns)


def _clamp(x, lo=1.0, hi=5.0):
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Mock 打分器
# --------------------------------------------------------------------------- #
class MockScorer:
    """规则打分器。返回每个指标 1-5 分及命中信号（用于报告可解释性）。"""

    name = "mock"

    def score(self, case):
        q = case.get("user_question", "")
        r = case.get("auto_reply", "")
        signals = []

        # ---- 有用性 ----
        helpful = 4.0
        escapes = _count_any(r, ESCAPE_HATCH_PATTERNS)
        # 用户已表达"搞不懂/太复杂"，回复却复述流程 -> 答非所问，属根本性失败，
        # 硬封顶有用性并跳过"可执行步骤"加分（此时步骤本身就是问题）。
        da_fei_suo_wen = (_has(q, ["搞半天", "太复杂", "都不知道怎么操作"]) and
                          bool(re.search(r"[1-9][\.\、]", r)))
        if da_fei_suo_wen:
            helpful = 1.8
            signals.append("用户卡住却复述流程(答非所问)")
        else:
            if escapes > 0:
                helpful -= 0.7 * min(escapes, 2)
                signals.append(f"甩锅话术({escapes})")
            if _has(r, PUSH_TO_PAGE_PATTERNS):
                # 若同时提供主动帮助，"也可自查详情页"属合理补充而非甩锅，从轻处罚
                push_pen = 0.8 if _has(r, PROACTIVE_PATTERNS) else 1.3
                helpful -= push_pen
                signals.append("推用户自查详情页")
            if _has(r, PATIENCE_PATTERNS):
                helpful -= 0.6
                signals.append("让用户耐心等待")
            if _has(r, PROACTIVE_PATTERNS):
                helpful += 0.6
                signals.append("主动帮办")
            if bool(re.search(r"[1-9][\.\、].+[2-9][\.\、]", r)):
                helpful += 0.4
                signals.append("给出可执行步骤")
        helpful = _clamp(helpful)

        # ---- 针对性 ----
        spec = 3.5
        user_item = _has(q, ["这个", "这款", "那款", "那两款", "那个", "我买的那个", "我要买的那个"])
        if user_item:
            if _has(r, ["这款", "该款", "此款"]) or _has(r, ["TPU", "100Wh", "20000mAh"]):
                spec += 0.8
            else:
                spec -= 1.0
                signals.append("用户指代具体商品但回复套模板")
        # 用户给出具体数字/场景
        ctx_match = False
        if _has(q, ["才买三天", "7天", "三天"]) and ("7天" in r or "质保期" in r):
            ctx_match = True
        if _has(q, ["两周", "用了两周"]) and ("30天" in r or "质保期" in r):
            ctx_match = True
        if _has(q, ["连续", "又是坏的", "上次"]) and ("连续" in r or "再次" in r or "加强" in r):
            ctx_match = True
        if _has(q, ["等了20分钟", "等了"]) and ("久等" in r or "抱歉" in r):
            ctx_match = True
        if ctx_match:
            spec += 0.5
        else:
            if _has(q, ["才买三天", "两周", "连续", "等了20分钟", "两天没更新"]):
                spec -= 0.6
                signals.append("未利用用户给定的上下文")
        # 一条消息多个问题
        if _has(q, ["顺便", "顺便看看", "两件事"]) or q.count("？") >= 2 or q.count("?") >= 2:
            if _has(r, ["1.", "1、", "两个问题", "分别"]):
                spec += 0.5
            else:
                spec -= 0.8
                signals.append("多问题未逐一回应")
        spec = _clamp(spec)

        # ---- 语气与共情 ----
        tone = 3.5
        if "您好" in r:
            tone += 0.3
        if _has(r, ["非常抱歉", "抱歉给您", "抱歉让您", "抱歉"]):
            tone += 0.5
        if "感谢" in r:
            tone += 0.3
        user_upset = _has(q, USER_FRUSTRATION_PATTERNS)
        if user_upset:
            if _has(r, ["非常抱歉", "抱歉让您", "抱歉给您"]):
                tone += 0.3
            if _has(r, INTERNAL_TALK_PATTERNS) and not _has(r, PROACTIVE_PATTERNS):
                tone -= 0.8
                signals.append("用户不满时谈内部事项")
        # 答非所问场景下，礼貌道歉是空话，削弱语气分
        if da_fei_suo_wen:
            tone -= 0.5
        tone -= 0.1 * min(escapes, 2)
        tone = _clamp(tone)

        # ---- 事实准确性 ----
        # 规则无法真正核实事实，仅据"是否给出可核对的具体信息"小幅调整；
        # 本数据集无明显臆造，故该维度普遍偏高，这也是 mock 的已知局限。
        acc = 4.0
        if _has(r, ["100Wh", "1-3个工作日", "3-7个工作日", "5-15个工作日",
                    "30天质保期", "7天质保期", "20000mAh", "74Wh", "TPU软胶"]):
            acc += 0.5
            signals.append("给出可核对的具体信息")
        if _has(r, ["一般情况下", "建议您", "可能"]) and not _has(r, ["100Wh", "30天"]):
            acc += 0.0
        acc = _clamp(acc)

        scores = {
            "accuracy": round(acc, 2),
            "helpfulness": round(helpful, 2),
            "specificity": round(spec, 2),
            "tone": round(tone, 2),
        }
        return scores, signals


# --------------------------------------------------------------------------- #
# LLM 打分器（可选）
# --------------------------------------------------------------------------- #
LLM_PROMPT_TEMPLATE = """你是客服自动回复质量评估员。请对下面这条自动回复打分。

用户问题：{question}
自动回复：{reply}

按以下 4 个指标各打 1-5 分（整数或一位小数），并给出一句理由：
- accuracy 事实准确性（5 无臆造且准确；1 关键信息错误/瞎编）
- helpfulness 有用性（5 直接解决问题；1 答非所问/让用户自己想办法）
- specificity 针对性（5 紧扣用户具体场景；1 套话模板）
- tone 语气共情（5 礼貌且共情；1 语气不当）

仅返回 JSON，格式：
{{"accuracy":4.0,"helpfulness":3.0,"specificity":3.0,"tone":4.0,"reason":"..."}}
"""


class LLMScorer:
    """调用 OpenAI 兼容 /v1/chat/completions 接口打分，支持并发。"""

    name = "llm"

    def __init__(self, max_workers=5, timeout=30):
        # 支持通用 OpenAI 兼容接口：DeepSeek / SiliconFlow / OpenAI 等
        self.api_key = (os.getenv("OPENAI_API_KEY") or
                        os.getenv("SILICONFLOW_API_KEY") or
                        os.getenv("DEEPSEEK_API_KEY"))
        self.base_url = (os.getenv("OPENAI_BASE_URL") or
                         os.getenv("SILICONFLOW_BASE_URL") or
                         os.getenv("DEEPSEEK_BASE_URL") or
                         "https://api.deepseek.com/v1")
        self.model = os.getenv("EVAL_LLM_MODEL") or "deepseek-chat"
        self.max_workers = max_workers
        self.timeout = timeout

    @property
    def available(self):
        return bool(self.api_key)

    def score(self, case):
        """单条打分（兼容旧接口）。"""
        return self._score_one(case)

    def score_batch(self, cases, fallback_scorer=None, progress_callback=None):
        """批量并发打分，单条失败可回退 fallback_scorer。

        cases: list[dict]
        fallback_scorer: 失败时使用的备用打分器，默认 MockScorer
        progress_callback: fn(case_id, status) 用于进度打印
        """
        if fallback_scorer is None:
            fallback_scorer = MockScorer()

        results = {case["id"]: None for case in cases}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {
                executor.submit(self._score_one, case): case["id"]
                for case in cases
            }
            for future in as_completed(future_to_id):
                cid = future_to_id[future]
                try:
                    scores, signals = future.result()
                    results[cid] = {"scores": scores, "signals": signals, "mode": "llm"}
                    if progress_callback:
                        progress_callback(cid, "ok")
                except Exception as e:
                    if progress_callback:
                        progress_callback(cid, f"fail({e})")
                    scores, signals = fallback_scorer.score(
                        next(c for c in cases if c["id"] == cid)
                    )
                    results[cid] = {"scores": scores, "signals": signals, "mode": "mock"}
        return [results[case["id"]] for case in cases]

    def _score_one(self, case):
        prompt = LLM_PROMPT_TEMPLATE.format(
            question=case.get("user_question", ""),
            reply=case.get("auto_reply", ""),
        )
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }).encode("utf-8")
        url = self.base_url.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        obj = _extract_json(content)
        scores = {k: float(obj.get(k, 3.0)) for k in METRICS.keys()}
        scores = {k: _clamp(v) for k, v in scores.items()}
        return scores, [obj.get("reason", "")]


def _extract_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #
def get_scorer(force_mock=False, max_workers=5, timeout=30):
    """优先使用 LLM；未配置 key 或 force_mock 时回退到 mock。"""
    if not force_mock:
        llm = LLMScorer(max_workers=max_workers, timeout=timeout)
        if llm.available:
            return llm
    return MockScorer()


def weighted_total(scores):
    """按 WEIGHTS 计算加权总分（1-5）。"""
    return round(sum(scores[k] * WEIGHTS[k] for k in WEIGHTS), 2)
