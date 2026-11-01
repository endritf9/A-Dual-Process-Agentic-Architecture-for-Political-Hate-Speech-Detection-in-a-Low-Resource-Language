import json
import re
import time


import config
from entity_kb import load_kb
from llm_client import LLMError, chat


SYSTEM_PROMPT = """You are a content moderation agent for Albanian-language social media comments \
(from Albania, Kosovo and North Macedonia; standard Albanian and Gheg, often written without diacritics, \
with :USER: and :URL: as placeholders).

Your task is to decide if a comment is POLITICAL HATE SPEECH (label 1) or NOT (label 0).

Label 1 - political hate speech: the comment shows hostility or contempt toward a political target (a \
politician, a party or its members, the government or another state institution, or a group defined in \
political or national terms). This includes insults, vulgar language, slurs, derogatory nicknames for \
politicians, mocking or sarcastic attacks, dehumanising words, accusations used as insults (e.g. "thief", \
"criminal", "traitor"), threats, wishes of harm, and calls for hostility or violence. The tone can be informal \
and the insult can be short or indirect.
Label 0 - not political hate speech: neutral mentions of politicians or institutions, news-like statements, \
questions, support or praise, and criticism of decisions or policies that is expressed without insults, \
mockery or contempt. Comments with no political target are also label 0.

Important: mentioning a politician, a party, an institution or a country is NOT hate speech by itself. \
Decide on what the comment says about the target, not on which names it contains.

{tool_text}When you are ready, call submit_decision with the label (0 or 1), your confidence (0-1) and a \
short rationale (one or two sentences, in English)."""

TOOL_TEXT = """You can use the tools below. Use them only when they help; for clear cases you can decide directly.
{tool_list}
"""

TOOL_SPECS = {
    "get_classifier_score": {
        "description": "Probabilities that the comment is political hate speech from the fast System 1 "
                       "classifiers (a fine-tuned mBERT and the zero-shot model Jev, with Jev's guess of the "
                       "target and a hostility score from 0 to 4). They are fast but can be wrong, and they "
                       "can disagree with each other.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "lookup_political_entities": {
        "description": "Finds political actors, parties, institutions, places and ethnic terms mentioned in the "
                       "comment and returns who or what they are. Optionally pass one word or name to look up.",
        "parameters": {"type": "object", "properties": {
            "term": {"type": "string", "description": "optional word or name to look up; empty = scan the comment"}},
            "required": []},
    },
    "get_thread_context": {
        "description": "Returns the comment this one replies to (if it is a reply) and up to three other comments "
                       "from the same thread. Useful when the comment is short, uses pronouns, or is unclear alone.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
SUBMIT_SPEC = {
    "type": "function",
    "function": {
        "name": "submit_decision",
        "description": "Submit the final decision for the comment.",
        "parameters": {"type": "object", "properties": {
            "label": {"type": "integer", "enum": [0, 1], "description": "1 = political hate speech, 0 = not"},
            "confidence": {"type": "number", "description": "confidence in the label, 0 to 1"},
            "rationale": {"type": "string", "description": "one or two sentences"}},
            "required": ["label", "confidence", "rationale"]},
    },
}

CONFIGS = {
    "zero_shot": {"tools": [], "few_shot": False},
    "few_shot": {"tools": [], "few_shot": True},
    "agent_full": {"tools": ["get_classifier_score", "lookup_political_entities", "get_thread_context"], "few_shot": False},
    "agent_no_classifier": {"tools": ["lookup_political_entities", "get_thread_context"], "few_shot": False},
    "agent_no_entities": {"tools": ["get_classifier_score", "get_thread_context"], "few_shot": False},
    "agent_no_context": {"tools": ["get_classifier_score", "lookup_political_entities"], "few_shot": False},
    
    
    "single_all": {"tools": [], "few_shot": False, "evidence": True},
    "single_no_classifier": {"tools": [], "few_shot": False, "evidence": True, "evidence_tools":
                             ["lookup_political_entities", "get_thread_context"]},
}
EVIDENCE_TOOLS = ["get_classifier_score", "lookup_political_entities", "get_thread_context"]


def _clip(s, n=400):
    s = str(s or "")
    return s if len(s) <= n else s[:n] + " ..."


class ToolBox:
    

    def __init__(self, case: dict):
        self.case = case
        self.kb = load_kb()

    def run(self, name, args):
        if name == "get_classifier_score":
            sc = self.case.get("classifier_scores") or {}
            if not sc:
                return {"error": "no classifier score available"}
            out = []
            for clf, e in sc.items():
                item = {"classifier": config.CLASSIFIER_DESCRIPTIONS.get(clf, clf),
                        "probability_political_hate_speech": round(float(e["p"]), 3)}
                if "target" in e:
                    item["likely_target"] = e["target"]
                if "hostility" in e:
                    item["hostility_0_to_4"] = e["hostility"]
                out.append(item)
            return {"system1_scores": out}
        if name == "lookup_political_entities":
            term = (args or {}).get("term") or ""
            hits = self.kb.lookup(term) if term.strip() else self.kb.match(self.case["text"])
            return {"entities": hits} if hits else {"entities": [], "note": "no known political entity found"}
        if name == "get_thread_context":
            related = json.loads(self.case.get("related_texts") or "[]")
            out = {"is_reply": bool(self.case.get("is_reply", 0))}
            if self.case.get("parent_text"):
                out["replies_to"] = _clip(self.case["parent_text"])
            if related:
                out["other_comments_in_thread"] = [_clip(r, 200) for r in related]
            if len(out) == 1:
                out["note"] = "no thread context available"
            return out
        return {"error": f"unknown tool {name}"}


def build_tools(names):
    tools = [{"type": "function", "function": {"name": n, **TOOL_SPECS[n]}} for n in names]
    return tools + [SUBMIT_SPEC]


def build_messages(case, cfg, shots=None):
    if cfg["tools"]:
        tl = "\n".join(f"- {n}: {TOOL_SPECS[n]['description']}" for n in cfg["tools"])
        tool_text = TOOL_TEXT.format(tool_list=tl) + "\n"
    else:
        tool_text = ""
    sys_msg = SYSTEM_PROMPT.format(tool_text=tool_text)
    user = ""
    if cfg["few_shot"] and shots:
        user += "Labelled examples:\n"
        for s in shots:
            user += f'- "{_clip(s["text"], 300)}" -> label {s["label"]}\n'
        user += "\n"
    user += f"Comment to classify:\n<<<\n{case['text']}\n>>>"
    if cfg.get("evidence"):
        box = ToolBox(case)
        ev = {n: box.run(n, {}) for n in cfg.get("evidence_tools", EVIDENCE_TOOLS)}
        user += ("\n\nAdditional information about this comment (outputs of the same tools that the agent version can "
                 "call; they can be wrong):\n" + json.dumps(ev, ensure_ascii=False, indent=1))
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]


def _parse_decision(obj):
    try:
        if isinstance(obj, str):
            m = re.search(r"\{.*\}", obj, re.S)
            obj = json.loads(m.group(0)) if m else {}
        label = int(obj.get("label"))
        if label not in (0, 1):
            return None
        conf = obj.get("confidence", None)
        return {"label": label, "confidence": float(conf) if conf is not None else None,
                "rationale": str(obj.get("rationale", ""))[:500]}
    except Exception:
        return None


NO_FORCED_TOOL = set()


def _chat(model_id, messages, tools, tool_choice):
    if model_id in NO_FORCED_TOOL and tool_choice != "auto":
        tool_choice = "auto"
    try:
        return chat(model_id, messages, tools=tools, tool_choice=tool_choice, extra=config_extra(model_id))
    except LLMError as e:
        if "tool_choice" in str(e) and tool_choice != "auto":
            NO_FORCED_TOOL.add(model_id)
            return chat(model_id, messages, tools=tools, tool_choice="auto", extra=config_extra(model_id))
        raise


def run_case(case, config_name, model_id, shots=None):
    
    cfg = CONFIGS[config_name]
    tools = build_tools(cfg["tools"])
    messages = build_messages(case, cfg, shots)
    box = ToolBox(case)
    rec = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "latency_s": 0.0, "llm_calls": 0,
           "tool_calls": [], "decision": None, "error": None, "cached_calls": 0}
    t_wall = time.perf_counter()
    steps = 0
    while True:
        force = (not cfg["tools"]) or steps >= config.AGENT_MAX_STEPS
        tool_choice = {"type": "function", "function": {"name": "submit_decision"}} if force else "auto"
        try:
            resp = _chat(model_id, messages, tools, tool_choice)
        except LLMError as e:
            rec["error"] = str(e)[:500]
            break
        rec["llm_calls"] += 1
        rec["cached_calls"] += int(resp.get("cached", False))
        for k in ("prompt_tokens", "completion_tokens", "cost"):
            rec[k] += resp["usage"].get(k, 0) or 0
        rec["latency_s"] += resp.get("latency_s", 0.0)
        msg = resp["message"] or {}
        tcs = msg.get("tool_calls") or []
        if not tcs:
            dec = _parse_decision(msg.get("content") or "")
            if dec:
                rec["decision"] = dec
                break
            if force and steps > config.AGENT_MAX_STEPS:
                rec["error"] = "no decision"
                break
            messages = messages + [{"role": "assistant", "content": msg.get("content") or ""},
                                   {"role": "user", "content": "Please call submit_decision now."}]
            steps += 1
            continue
        messages = messages + [{"role": "assistant", "content": msg.get("content") or "", "tool_calls": [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or "{}"}}
            for tc in tcs]}]
        done = False
        for tc in tcs:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "submit_decision":
                rec["decision"] = _parse_decision(args)
                done = rec["decision"] is not None
                result = {"ok": done}
            else:
                result = box.run(name, args)
                rec["tool_calls"].append(name)
            messages = messages + [{"role": "tool", "tool_call_id": tc["id"], "name": name,
                                    "content": json.dumps(result, ensure_ascii=False)}]
        if done:
            break
        steps += 1
        if steps > config.AGENT_MAX_STEPS + 1:
            rec["error"] = "too many steps"
            break
    rec["wall_s"] = time.perf_counter() - t_wall
    return rec


def config_extra(model_id):
    return config.MODEL_EXTRA.get(model_id, config.DEFAULT_EXTRA)
