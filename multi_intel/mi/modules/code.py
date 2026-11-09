"""Code/computation module. Psychological seat: procedural skill + verifiability.

This is the module whose cues are best of all, because its output can be *checked*
rather than *estimated*: an expression either parses and evaluates or it does not.
Where verification exists, the system should prefer it over any confidence guess.
"""
from __future__ import annotations

import ast
import operator as op
import re

from ..types import Cues, Result, Stimulus
from .base import Module

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Pow: op.pow,
        ast.USub: op.neg, ast.FloorDiv: op.floordiv, ast.Mod: op.mod}
_EXPR = re.compile(r"-?\d+(?:\s*[+\-*%]\s*-?\d+|\s*\*\*\s*\d+|\s*//\s*-?\d+)+")


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_safe_eval(node.right)) > 64:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported")


class CodeModule(Module):
    name = "code"
    cue_names = ("parseable", "simplicity")

    def _extract(self, s: Stimulus):
        m = _EXPR.search(s.text)
        return m.group(0) if m else None

    def cues(self, s: Stimulus) -> Cues:
        expr = self._extract(s)
        if not expr:
            return Cues(values={"parseable": 0.0, "simplicity": 0.0}, cost=0.0)
        try:
            ast.parse(expr, mode="eval")
            ok = 1.0
        except SyntaxError:
            ok = 0.0
        return Cues(values={"parseable": ok, "simplicity": max(0.0, 1.0 - len(expr) / 60.0)}, cost=0.0)

    def run(self, s: Stimulus) -> Result:
        expr = self._extract(s)
        if not expr:
            return Result(module=self.name, answer=None, verified=False, detail={"reason": "no expression"})
        try:
            val = _safe_eval(ast.parse(expr, mode="eval"))
            return Result(module=self.name, answer=val, verified=True, detail={"expr": expr})
        except Exception as e:  # noqa: BLE001
            return Result(module=self.name, answer=None, verified=False, detail={"error": str(e)})
