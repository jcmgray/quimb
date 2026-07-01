"""Parsing of qsim and OpenQASM 2/3 into circuit gate lists."""

import ast
import copy
import functools
import math
import numbers
import operator
import re
import warnings

import numpy as np
from autoray import (
    do,
)

from ...utils import (
    concatv,
    partitionby,
)
from .gates import ALL_GATES, Gate


def _convert_ints_and_floats(x):
    if isinstance(x, str):
        try:
            return int(x)
        except ValueError:
            pass

        try:
            return float(x)
        except ValueError:
            pass

    return x


def _put_registers_last(x):
    # no need to do anything unless parameter (i.e. float) is found last
    if not isinstance(x[-1], float):
        return x

    # swap this last group of floats with the penultimate group of integers
    parts = tuple(partitionby(type, x))
    return tuple(concatv(*parts[:-2], parts[-1], parts[-2]))


def parse_qsim_str(contents):
    """Parse a 'qsim' input format string into circuit information.

    The format is described here: https://quantumai.google/qsim/input_format.

    Parameters
    ----------
    contents : str
        The full string of the qsim file.

    Returns
    -------
    circuit_info : dict
        Information about the circuit:

        - circuit_info['n']: the number of qubits
        - circuit_info['n_gates']: the number of gates in total
        - circuit_info['gates']: list[list[str]], list of gates, each of which
          is a list of strings read from a line of the qsim file.
    """

    lines = contents.split("\n")
    n = int(lines[0])

    # turn into tuples of python types
    gates = [
        tuple(map(_convert_ints_and_floats, line.strip().split(" ")))
        for line in lines[1:]
        if line
    ]

    # put registers/parameters in standard order and detect if gate round used
    gates = tuple(map(_put_registers_last, gates))
    round_specified = isinstance(gates[0][0], numbers.Integral)

    return {
        "n": n,
        "gates": gates,
        "n_gates": len(gates),
        "round_specified": round_specified,
    }


def parse_qsim_file(fname, **kwargs):
    """Parse a qsim file."""
    with open(fname) as f:
        return parse_qsim_str(f.read(), **kwargs)


def parse_qsim_url(url, **kwargs):
    """Parse a qsim url."""
    from urllib import request

    return parse_qsim_str(request.urlopen(url).read().decode(), **kwargs)


def to_clean_list(s, delimiter):
    """Split, strip and filter a string by a given character into a list."""
    if s is None:
        return []
    return list(filter(None, (w.strip() for w in s.split(delimiter))))


def multi_replace(s, replacements):
    """Replace multiple substrings in a string."""
    for w, r in replacements.items():
        s = s.replace(w, r)
    return s


def _openqasm_replace_tokens(s, replacements):
    """Replace whole identifier-like tokens in an OpenQASM fragment."""
    if not replacements:
        return s

    pattern = "|".join(
        sorted(map(re.escape, replacements), key=len, reverse=True)
    )
    return re.sub(
        rf"(?<!\w)({pattern})(?!\w)",
        lambda match: replacements[match.group(1)],
        s,
    )


@functools.lru_cache(None)
def get_openqasm2_regexes():
    return {
        "header": re.compile(r"(OPENQASM\s+2.0;)|(include\s+\"qelib1.inc\";)"),
        "qreg": re.compile(r"qreg\s+(\w+)\s*\[(\d+)\];"),
        "gate": re.compile(r"(\w+)\s*(\((.+)\))?\s*(.*);"),
        "error": re.compile(r"^(reset|if|for)\b"),
        "ignore": re.compile(r"^(creg|measure|barrier)\b"),
        "gate_def": re.compile(r"^gate\s+"),
        "gate_sig": re.compile(r"^gate\s+(\w+)\s*(\((.+)\))?\s*(.*)"),
    }


@functools.lru_cache(None)
def get_openqasm3_regexes():
    return {
        "header": re.compile(
            r"(OPENQASM\s+3(?:\.\d+)?;)"
            r"|(include\s+\"(?:stdgates|qelib1)\.inc\";)"
        ),
        "qubit": re.compile(r"qubit(?:\s*\[(.+)\])?\s+(\w+);"),
        "input": re.compile(r"input\s+\w+(?:\s*\[[^\]]+\])?\s+(\w+);"),
        "output": re.compile(r"output\s+\w+(?:\s*\[[^\]]+\])?\s+(\w+);"),
        "const": re.compile(
            r"const\s+\w+(?:\s*\[[^\]]+\])?\s+(\w+)\s*=\s*(.+);"
        ),
        "classical_decl": re.compile(
            r"(bit|bool|int|uint|float|angle|complex)(?:\s*\[[^\]]+\])?\s+"
            r"(\w+)(?:\s*=\s*(.+))?;"
        ),
        "array_decl": re.compile(r"array\s*\[.*\]\s+(\w+)\s*=\s*(.+);"),
        "assign": re.compile(r"(\w+(?:\s*\[[^\]]+\])?)\s*=\s*(.+);"),
        "ignore": re.compile(r"^(measure|barrier|gphase)\b"),
        "error": re.compile(
            r"^(reset|if|for|while|switch|box|delay|defcal|cal|extern|pragma"
            r"|alias|return)\b"
        ),
        "gate_def": re.compile(r"^gate\s+"),
        "gate_sig": re.compile(r"^gate\s+(\w+)\s*(?:\((.*?)\))?\s*(.*?)\s*$"),
        "gate": re.compile(r"(\w+)\s*(?:\((.*)\))?\s*(.*);"),
    }


def _openqasm_split_top_level(s, sep=","):
    if not s:
        return []

    parts = []
    cur = []
    depth_paren = 0
    depth_brack = 0
    depth_brace = 0

    for c in s:
        if c == "(":
            depth_paren += 1
        elif c == ")":
            depth_paren -= 1
        elif c == "[":
            depth_brack += 1
        elif c == "]":
            depth_brack -= 1
        elif c == "{":
            depth_brace += 1
        elif c == "}":
            depth_brace -= 1

        if (
            c == sep
            and depth_paren == 0
            and depth_brack == 0
            and depth_brace == 0
        ):
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)

    if cur:
        parts.append("".join(cur).strip())

    return [p for p in parts if p]


def _strip_openqasm_comments(contents):
    """Remove ``//`` line comments and ``/* ... */`` block comments from an
    OpenQASM source string, including ones that begin or end part way through a
    line. Comment markers inside double-quoted strings are ignored, and
    newlines are preserved so line numbers are retained.
    """
    out = []
    i = 0
    n = len(contents)
    in_block = in_line = in_string = False
    while i < n:
        c = contents[i]
        pair = contents[i : i + 2]
        if in_block:
            if pair == "*/":
                in_block = False
                i += 2
            else:
                if c == "\n":
                    out.append(c)
                i += 1
        elif in_line:
            if c == "\n":
                in_line = False
                out.append(c)
            i += 1
        elif in_string:
            out.append(c)
            in_string = c != '"'
            i += 1
        elif pair == "/*":
            in_block = True
            i += 2
        elif pair == "//":
            in_line = True
            i += 2
        else:
            in_string = c == '"'
            out.append(c)
            i += 1
    return "".join(out)


def _split_openqasm_statements(contents):
    """Split an OpenQASM source string into individual statements, breaking on
    top-level ``;`` and after the closing ``}`` of a top-level block such as a
    gate definition. Bracketed and quoted regions are kept intact, so multiple
    statements on one physical line, or a single statement spread across lines,
    are both handled.
    """
    statements = []
    cur = []
    depth = 0
    in_string = False
    i = 0
    n = len(contents)
    while i < n:
        c = contents[i]
        if in_string:
            cur.append(c)
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
        elif c in "([{":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "}":
            depth -= 1
            cur.append(c)
            i += 1
            if depth == 0:
                # a block such as a gate definition ends at this '}', unless
                # it is part of an expression terminated by ';' (e.g. an array
                # literal initializer)
                j = i
                while j < n and contents[j].isspace():
                    j += 1
                if j >= n or contents[j] != ";":
                    statements.append("".join(cur).strip())
                    cur = []
            continue
        cur.append(c)
        if c == ";" and depth == 0:
            statements.append("".join(cur).strip())
            cur = []
        i += 1

    tail = "".join(cur).strip()
    if tail:
        statements.append(tail)

    return [s for s in statements if s]


def _openqasm_eval_expr(expr, env):
    if callable(expr):
        return expr(env)

    if not isinstance(expr, str):
        return expr

    expr = expr.strip()
    if not expr:
        return None

    tree = ast.parse(expr, mode="eval")

    allowed_binary_ops = {
        ast.Add: ("+", operator.add),
        ast.Sub: ("-", operator.sub),
        ast.Mult: ("*", operator.mul),
        ast.Div: ("/", operator.truediv),
        ast.FloorDiv: ("//", operator.floordiv),
        ast.Mod: ("%", operator.mod),
        ast.Pow: ("**", operator.pow),
        ast.LShift: ("<<", operator.lshift),
        ast.RShift: (">>", operator.rshift),
        ast.BitAnd: ("&", operator.and_),
        ast.BitXor: ("^", operator.xor),
        ast.BitOr: ("|", operator.or_),
    }
    allowed_unary_ops = {
        ast.USub: ("-", operator.neg),
        ast.UAdd: ("+", operator.pos),
        ast.Invert: ("~", operator.invert),
        ast.Not: ("!", operator.not_),
    }
    allowed_fns = {
        "sin": lambda x: do("sin", x),
        "cos": lambda x: do("cos", x),
        "tan": lambda x: do("tan", x),
        "asin": lambda x: do("arcsin", x),
        "acos": lambda x: do("arccos", x),
        "atan": lambda x: do("arctan", x),
        "exp": lambda x: do("exp", x),
        "ln": lambda x: do("log", x),
        "log": lambda x: do("log", x),
        "sqrt": lambda x: do("sqrt", x),
        "abs": lambda x: do("abs", x),
        "pow": pow,
    }

    def _placeholder_passthrough(*xs):
        for x in xs:
            if _is_interface_placeholder(x):
                return x
        raise TypeError("No placeholder value supplied.")

    def _is_symbolic(x):
        if isinstance(x, str):
            return True
        if isinstance(x, (list, tuple)):
            return any(_is_symbolic(xi) for xi in x)
        return False

    def _combine_symbolic(op, *xs):

        def fmt(x):
            return x if isinstance(x, str) else repr(x)

        if len(xs) == 1:
            (x,) = xs
            if op == "+":
                return f"(+{fmt(x)})"
            if op == "-":
                return f"(-{fmt(x)})"
            if op == "~":
                return f"(~{fmt(x)})"
            if op == "!":
                return f"(!{fmt(x)})"
        if op in {
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "exp",
            "ln",
            "log",
            "sqrt",
            "abs",
        }:
            return f"{op}({', '.join(map(fmt, xs))})"
        return f"({fmt(xs[0])} {op} {fmt(xs[1])})"

    allowed_compare_ops = {
        ast.Eq: ("==", operator.eq),
        ast.NotEq: ("!=", operator.ne),
        ast.Lt: ("<", operator.lt),
        ast.LtE: ("<=", operator.le),
        ast.Gt: (">", operator.gt),
        ast.GtE: (">=", operator.ge),
    }

    def _eval_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return math.pi
            if node.id in env:
                return env[node.id]
            raise NotImplementedError(
                f"Unknown OpenQASM 3 identifier: {node.id}"
            )
        if isinstance(node, ast.BinOp):
            lhs = _eval_node(node.left)
            rhs = _eval_node(node.right)
            optype = type(node.op)
            if optype not in allowed_binary_ops:
                raise NotImplementedError(
                    f"Unsupported OpenQASM 3 operator: {optype.__name__}"
                )
            op, fn = allowed_binary_ops[optype]
            if _is_symbolic(lhs) or _is_symbolic(rhs):
                return _combine_symbolic(op, lhs, rhs)
            if _is_interface_placeholder(lhs) or _is_interface_placeholder(
                rhs
            ):
                return _placeholder_passthrough(lhs, rhs)
            return fn(lhs, rhs)
        if isinstance(node, ast.UnaryOp):
            x = _eval_node(node.operand)
            optype = type(node.op)
            if optype not in allowed_unary_ops:
                raise NotImplementedError(
                    f"Unsupported OpenQASM 3 unary op: {optype.__name__}"
                )
            op, fn = allowed_unary_ops[optype]
            if _is_symbolic(x):
                return _combine_symbolic(op, x)
            if _is_interface_placeholder(x):
                return x
            return fn(x)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise NotImplementedError(
                    "Unsupported OpenQASM 3 callable expression."
                )
            name = node.func.id
            if name not in allowed_fns:
                raise NotImplementedError(
                    f"Unsupported OpenQASM 3 function: {name}"
                )
            args = [_eval_node(a) for a in node.args]
            if any(_is_symbolic(a) for a in args):
                return _combine_symbolic(name, *args)
            if any(_is_interface_placeholder(a) for a in args):
                return _placeholder_passthrough(*args)
            return allowed_fns[name](*args)
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise NotImplementedError(
                    "Chained comparisons not supported in OpenQASM 3."
                )
            lhs = _eval_node(node.left)
            rhs = _eval_node(node.comparators[0])
            optype = type(node.ops[0])
            if optype not in allowed_compare_ops:
                raise NotImplementedError(
                    f"Unsupported OpenQASM 3 compare op: {optype.__name__}"
                )
            op, fn = allowed_compare_ops[optype]
            if _is_symbolic(lhs) or _is_symbolic(rhs):
                return _combine_symbolic(op, lhs, rhs)
            if _is_interface_placeholder(lhs) or _is_interface_placeholder(
                rhs
            ):
                return _placeholder_passthrough(lhs, rhs)
            return fn(lhs, rhs)
        if isinstance(node, ast.List):
            return [_eval_node(x) for x in node.elts]
        if isinstance(node, ast.Subscript):
            value = _eval_node(node.value)
            index_node = node.slice
            if hasattr(ast, "Index") and isinstance(index_node, ast.Index):
                index_node = index_node.value
            index = _eval_node(index_node)
            if isinstance(index, numbers.Number):
                index = int(index)
            else:
                raise NotImplementedError(
                    "Symbolic array indices are unsupported."
                )
            if not isinstance(value, (list, tuple)):
                raise NotImplementedError(
                    "Only OpenQASM 3 array-style values can be indexed."
                )
            return value[index]
        raise NotImplementedError(
            f"Unsupported OpenQASM 3 expression node: {type(node).__name__}"
        )

    return _eval_node(tree.body)


def _is_interface_placeholder(x):
    try:
        from ..interface import Placeholder
    except ImportError:
        return False
    return isinstance(x, Placeholder)


def _placeholder_param_vector(values):
    from ..interface import Placeholder

    for value in values:
        if _is_interface_placeholder(value):
            dtype = getattr(value, "dtype", "float64")
            if dtype in (None, "unknown"):
                dtype = "float64"
            return Placeholder(np.empty((len(values),), dtype=dtype))

    raise TypeError("No placeholder values supplied.")


# mapping from lower-case OpenQASM gate names to canonical quimb gate labels,
# shared by the OpenQASM 2 and 3 parsers
OPENQASM_GATE_ALIASES = {
    "u": "U3",
    "u1": "U1",
    "u2": "U2",
    "u3": "U3",
    "p": "PHASE",
    "phase": "PHASE",
    "id": "IDEN",
    "i": "IDEN",
    "cnot": "CNOT",
    "cx": "CX",
    "cy": "CY",
    "cz": "CZ",
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "sdg": "SDG",
    "t": "T",
    "tdg": "TDG",
    "sx": "SX",
    "sxdg": "SXDG",
    "swap": "SWAP",
    "iswap": "ISWAP",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "crx": "CRX",
    "cry": "CRY",
    "crz": "CRZ",
    "cu1": "CU1",
    "cu2": "CU2",
    "cu3": "CU3",
    "cphase": "CPHASE",
    "cp": "CPHASE",
    "ccx": "CCX",
    "ccnot": "CCX",
    "toffoli": "CCX",
    "cswap": "CSWAP",
    "fredkin": "CSWAP",
}


def _resolve_qubit_arg(token, sitemap, registers, env=None):
    """Resolve a single OpenQASM qubit argument into either a qubit index or,
    for a whole-register reference, a tuple of indices to broadcast over.
    """
    env = {} if env is None else env
    token = token.strip()

    # name bound to a collection of indices in env, e.g. "q" -> [0, 1]
    if token in env and isinstance(env[token], (tuple, list)):
        return tuple(env[token])

    # whole register, e.g. "q" -> (0, 1, 2), a size-1 register -> bare index 0
    if token in registers:
        reg = registers[token]
        return reg if len(reg) > 1 else reg[0]

    # indexed element, e.g. "q[0]" or "q[n - 1]" with n resolved from env
    match = re.fullmatch(r"(\w+)\[(.+)\]", token)
    if match:
        base, idx_expr = match.groups()
        idx = _openqasm_eval_expr(idx_expr, env)
        # the index must be concrete, e.g. "q[theta]" is rejected
        if not isinstance(idx, numbers.Number):
            raise NotImplementedError(
                "Symbolic qubit indices are unsupported."
            )
        idx = int(idx)
        # index into an env collection, else the global sitemap "q[2]" -> 2
        if base in env and isinstance(env[base], (tuple, list)):
            return env[base][idx]
        return sitemap[f"{base}[{idx}]"]

    # not a register, env collection or indexed element, e.g. "foo"
    raise NotImplementedError(f"Unknown qubit identifier: {token}")


def _broadcast_gate_qubits(resolved):
    """Given the resolved qubit arguments of a gate (each either a single index
    or a register tuple), return the list of concrete qubit tuples to apply,
    broadcasting any whole-register arguments over their length.
    """
    sizes = {len(q) for q in resolved if isinstance(q, (tuple, list))}
    if not sizes:
        return [tuple(resolved)]
    if len(sizes) != 1:
        raise NotImplementedError(
            "Broadcasted gate args must use registers of equal length."
        )
    (size,) = sizes
    return [
        tuple(
            value[i] if isinstance(value, (tuple, list)) else value
            for value in resolved
        )
        for i in range(size)
    ]


def parse_openqasm2_str(contents):
    """Parse the string contents of an OpenQASM 2.0 file. This shares the gate
    aliasing, arithmetic expression evaluation and whole-register broadcasting
    of the OpenQASM 3 parser. It does not support classical control flow and is
    not guaranteed to check the full OpenQASM grammar.
    """
    # define regular expressions for parsing
    rgxs = get_openqasm2_regexes()

    # strip comments and normalize to one statement per line so that inline
    # comments, code following a block comment, and several statements on one
    # line are all handled
    contents = _strip_openqasm_comments(contents)
    contents = "\n".join(_split_openqasm_statements(contents))

    # initialise number of qubits to zero and an empty list for gates
    sitemap = {}
    registers = {}
    gates = []
    custom_gates = {}
    # openqasm 2 has no symbolic parameters, but an empty env lets the shared
    # expression and qubit-resolution helpers be reused
    env = {}
    # only want to warn once about each ignored instruction
    warned = {}

    # Process each line
    lines = contents.split("\n")
    while lines:
        line = lines.pop(0).strip()
        if not line:
            # blank line
            continue
        if rgxs["header"].match(line):
            # ignore standard header lines
            continue

        match = rgxs["qreg"].match(line)
        if match:
            # quantum register -> extend sites
            name, nq = match.groups()
            registers[name] = tuple(
                range(len(sitemap), len(sitemap) + int(nq))
            )
            for i, q in enumerate(registers[name]):
                sitemap[f"{name}[{i}]"] = q
            continue

        match = rgxs["ignore"].match(line)
        if match:
            # certain operations we can just ignore and warn about
            (op,) = match.groups()
            if not warned.get(op, False):
                warnings.warn(
                    f"Unsupported operation ignored: {op}", SyntaxWarning
                )
                warned[op] = True
            continue

        if rgxs["error"].match(line):
            # raise hard error for custom tate defns etc
            raise NotImplementedError(
                f"The following instruction is not supported: {line}"
            )

        if rgxs["gate_def"].match(line):
            # custom gate definition:
            # first gather all lines involved in the gate definition
            gate_lines = [line]
            while True:
                if "}" in line:
                    # finished -> break
                    break
                else:
                    # not finished -> need next line
                    line = lines.pop(0)
                    gate_lines.append(line)

            # then combine this full gate definition, without newlines
            gate_body = "".join(gate_lines)
            # separate the signature and body
            gate_sig, gate_body = re.match(
                r"(.*)\s*{(.*)}", gate_body
            ).groups()

            # parse the signature
            match = rgxs["gate_sig"].match(gate_sig)
            label = match[1]
            sig_params = to_clean_list(match[3], ",")
            sig_qubits = to_clean_list(match[4], ",")

            # break body only back into individual lines, include semicolons
            gate_body = to_clean_list(gate_body, ";")
            # insert formatters, (using simple `replace` on the whole line will
            # scramble the label if parameters or qubits are letters etc)
            for i, gate_line in enumerate(gate_body):
                gm = rgxs["gate"].match(gate_line + ";")
                glabel = gm[1]
                gqubits = multi_replace(
                    gm[4], {q: f"{{{q}}}" for q in sig_qubits}
                )
                if gm[3]:
                    # sub gate line is parametrized gate
                    gparams = multi_replace(
                        gm[3], {p: f"{{{p}}}" for p in sig_params}
                    )
                    gate_body[i] = f"{glabel}({gparams}) {gqubits};"
                else:
                    # sub gate line is standard gate
                    gate_body[i] = f"{glabel} {gqubits};"

            custom_gates[label] = sig_params, sig_qubits, gate_body
            continue

        match = rgxs["gate"].search(line)
        if match:
            # apply a gate
            label, params, qubits = (
                match.group(1),
                match.group(3),
                match.group(4),
            )

            if label in custom_gates:
                # custom gate -> resolve parameters and qubits and prepend
                # the constituent gate lines to the main list
                sig_params, sig_qubits, gate_body = custom_gates[label]
                replacer = {
                    **dict(zip(sig_params, to_clean_list(params, ","))),
                    **dict(zip(sig_qubits, to_clean_list(qubits, ","))),
                }

                # recurse by prepending the translated gate body
                for gl in reversed(gate_body):
                    lines.insert(0, gl.format(**replacer))

                continue

            # standard gate -> resolve aliases, parameters and qubits
            label = OPENQASM_GATE_ALIASES.get(label.lower(), label)
            if label not in ALL_GATES:
                raise NotImplementedError(f"Unknown gate: {label}")

            params = tuple(
                _openqasm_eval_expr(p, env)
                for p in _openqasm_split_top_level(params)
            )
            resolved = [
                _resolve_qubit_arg(q, sitemap, registers, env)
                for q in _openqasm_split_top_level(qubits)
            ]
            for call_qubits in _broadcast_gate_qubits(resolved):
                gates.append(Gate(label, params, call_qubits))
            continue

        # if not covered by previous checks, simply raise
        raise SyntaxError(f"{line}")

    return {
        "n": len(sitemap),
        "sitemap": sitemap,
        "gates": gates,
        "n_gates": len(gates),
    }


def parse_openqasm2_file(fname, **kwargs):
    """Parse an OpenQASM 2.0 file."""
    with open(fname) as f:
        return parse_openqasm2_str(f.read(), **kwargs)


def parse_openqasm2_url(url, **kwargs):
    """Parse an OpenQASM 2.0 url."""
    from urllib import request

    return parse_openqasm2_str(request.urlopen(url).read().decode(), **kwargs)


def parse_openqasm3_str(contents):
    """Parse an OpenQASM 3.0 program from a string.

    This parser is dependency free and supports a practical subset of
    OpenQASM 3 for circuit import, including qubit declarations, input
    declarations, arithmetic expressions, custom gates, and register
    broadcasting.

    Parameters
    ----------
    contents : str
        The OpenQASM 3 source code to parse.

    Returns
    -------
    dict
        A dictionary describing the circuit with the following entries:

        - ``"n"``: total number of qubits.
        - ``"sitemap"``: mapping from OpenQASM qubit names to qubit indices.
        - ``"gates"``: parsed sequence of :class:`Gate` objects.
        - ``"n_gates"``: total number of parsed gates.
        - ``"inputs"``: tuple of symbolic input names declared with
          ``input``.
        - ``"symbols"``: mapping of symbolic names to their current values or
          symbolic placeholders.
        - ``"expressions"``: mapping from gate indices to symbolic parameter
          expressions requiring later binding.

    Raises
    ------
    NotImplementedError
        If the program uses unsupported OpenQASM 3 features such as control
        flow, calibration blocks, output declarations, or unsupported
        operations.
    SyntaxError
        If the source contains an instruction that does not match the
        supported grammar subset.
    """
    rgxs = get_openqasm3_regexes()
    # strip comments and normalize to one statement per line so that inline
    # comments, code following a block comment, and several statements on one
    # line are all handled
    contents = _strip_openqasm_comments(contents)
    contents = "\n".join(_split_openqasm_statements(contents))

    sitemap = {}
    registers = {}
    gates = []
    custom_gates = {}
    env = {}
    inputs = []
    symbols = {}
    expressions = {}
    warned = {}

    lines = contents.split("\n")
    while lines:
        line = lines.pop(0).strip()
        if not line:
            continue
        if rgxs["header"].match(line):
            continue

        match = rgxs["qubit"].match(line)
        if match:
            size_expr, name = match.groups()
            size = (
                1
                if size_expr is None
                else int(_openqasm_eval_expr(size_expr, env))
            )
            registers[name] = tuple(range(len(sitemap), len(sitemap) + size))
            for i, q in enumerate(registers[name]):
                sitemap[f"{name}[{i}]"] = q
            continue

        match = rgxs["input"].match(line)
        if match:
            (name,) = match.groups()
            inputs.append(name)
            env[name] = name
            symbols[name] = name
            continue

        if rgxs["output"].match(line):
            raise NotImplementedError("Output declarations are unsupported.")

        match = rgxs["const"].match(line)
        if match:
            name, expr = match.groups()
            env[name] = _openqasm_eval_expr(expr, env)
            continue

        match = rgxs["classical_decl"].match(line)
        if match:
            ctype, name, expr = match.groups()
            if ctype == "bit" and expr is None:
                if not warned.get("bit", False):
                    warnings.warn(
                        "Unsupported operation ignored: bit",
                        SyntaxWarning,
                    )
                    warned["bit"] = True
                continue
            if expr is not None:
                if expr.lstrip().startswith("measure "):
                    if not warned.get("measure", False):
                        warnings.warn(
                            "Unsupported operation ignored: measure",
                            SyntaxWarning,
                        )
                        warned["measure"] = True
                    continue
                env[name] = _openqasm_eval_expr(expr, env)
            continue

        match = rgxs["array_decl"].match(line)
        if match:
            name, expr = match.groups()
            env[name] = _openqasm_eval_expr(
                expr.replace("{", "[").replace("}", "]"), env
            )
            continue

        match = rgxs["assign"].match(line)
        if match:
            name, expr = match.groups()
            if expr.lstrip().startswith("measure "):
                if not warned.get("measure", False):
                    warnings.warn(
                        "Unsupported operation ignored: measure",
                        SyntaxWarning,
                    )
                    warned["measure"] = True
                continue
            env[name] = _openqasm_eval_expr(expr, env)
            continue

        if rgxs["ignore"].match(line):
            op = rgxs["ignore"].match(line).group(1)
            if not warned.get(op, False):
                warnings.warn(
                    "Unsupported operation ignored: global phase"
                    if op == "gphase"
                    else f"Unsupported operation ignored: {op}",
                    SyntaxWarning,
                )
                warned[op] = True
            continue

        if rgxs["error"].match(line):
            raise NotImplementedError(
                f"The following instruction is not supported: {line}"
            )

        if "@" in line:
            raise NotImplementedError(
                f"The following instruction is not supported: {line}"
            )

        if rgxs["gate_def"].match(line):
            gate_lines = [line]
            brace_count = line.count("{") - line.count("}")
            while brace_count > 0:
                line = lines.pop(0)
                gate_lines.append(line)
                brace_count += line.count("{") - line.count("}")

            gate_def = " ".join(gl.strip() for gl in gate_lines)
            gate_sig, gate_body = re.match(r"(.*)\s*{(.*)}", gate_def).groups()
            match = rgxs["gate_sig"].match(gate_sig)
            label = match[1]
            sig_params = _openqasm_split_top_level(match[2])
            sig_qubits = _openqasm_split_top_level(match[3])
            gate_body = _openqasm_split_top_level(gate_body, ";")

            for i, gate_line in enumerate(gate_body):
                gm = rgxs["gate"].match(gate_line + ";")
                glabel = gm[1]
                gqubits = _openqasm_replace_tokens(
                    gm[3], {q: f"{{{q}}}" for q in sig_qubits}
                )
                if gm[2]:
                    gparams = _openqasm_replace_tokens(
                        gm[2], {p: f"{{{p}}}" for p in sig_params}
                    )
                    gate_body[i] = f"{glabel}({gparams}) {gqubits};"
                else:
                    gate_body[i] = f"{glabel} {gqubits};"

            custom_gates[label] = (sig_params, sig_qubits, gate_body)
            continue

        match = rgxs["gate"].match(line)
        if match:
            label = match[1]
            params = _openqasm_split_top_level(match[2])
            qubits = _openqasm_split_top_level(match[3])

            if label in custom_gates:
                sig_params, sig_qubits, gate_body = custom_gates[label]
                if len(sig_params) != len(params):
                    raise NotImplementedError(
                        f"Custom gate {label} expected "
                        f"{len(sig_params)} parameters, got {len(params)}"
                    )

                replacer_base = dict(zip(sig_params, params))
                resolved = [
                    _resolve_qubit_arg(q, sitemap, registers, env)
                    for q in qubits
                ]
                sizes = {
                    len(q) for q in resolved if isinstance(q, (tuple, list))
                }
                if not sizes:
                    qubit_calls = [tuple(qubits)]
                else:
                    if len(sizes) != 1:
                        raise NotImplementedError(
                            "Broadcasted gate args must use registers of "
                            "equal length."
                        )
                    (size,) = sizes
                    qubit_calls = [
                        tuple(
                            f"{token}[{i}]"
                            if isinstance(value, (tuple, list))
                            else token
                            for token, value in zip(qubits, resolved)
                        )
                        for i in range(size)
                    ]

                for call_qubits in reversed(qubit_calls):
                    if len(sig_qubits) != len(call_qubits):
                        raise NotImplementedError(
                            f"Custom gate {label} expected "
                            f"{len(sig_qubits)} qubits, "
                            f"got {len(call_qubits)}"
                        )
                    replacer = {
                        **replacer_base,
                        **dict(zip(sig_qubits, map(str, call_qubits))),
                    }
                    for gl in reversed(gate_body):
                        lines.insert(0, gl.format(**replacer))
                continue

            label = OPENQASM_GATE_ALIASES.get(label.lower(), label)
            if label not in ALL_GATES:
                raise NotImplementedError(f"Unknown gate: {label}")

            raw_params = tuple(_openqasm_eval_expr(p, env) for p in params)
            resolved = [
                _resolve_qubit_arg(q, sitemap, registers, env) for q in qubits
            ]
            qubit_calls = _broadcast_gate_qubits(resolved)
            parametrize = any(
                not isinstance(p, numbers.Number) for p in raw_params
            )
            params = tuple(
                float("nan") if not isinstance(p, numbers.Number) else p
                for p in raw_params
            )
            for call_qubits in qubit_calls:
                if parametrize:
                    expressions[len(gates)] = raw_params
                gates.append(
                    Gate(
                        label=label,
                        params=params,
                        qubits=call_qubits,
                        parametrize=parametrize,
                    )
                )
            continue

        raise SyntaxError(f"{line}")

    return {
        "n": len(sitemap),
        "sitemap": sitemap,
        "gates": gates,
        "n_gates": len(gates),
        "inputs": tuple(inputs),
        "symbols": copy.copy(symbols),
        "expressions": copy.copy(expressions),
    }


def parse_openqasm3_file(fname, **kwargs):
    """Parse an OpenQASM 3.0 file.

    Parameters
    ----------
    fname : str or path-like
        Path to the OpenQASM 3 file.
    **kwargs
        Forwarded to :func:`parse_openqasm3_str`.

    Returns
    -------
    dict
        The parsed circuit information returned by
        :func:`parse_openqasm3_str`.
    """
    with open(fname) as f:
        return parse_openqasm3_str(f.read(), **kwargs)


def parse_openqasm3_url(url, **kwargs):
    """Parse an OpenQASM 3.0 program from a URL.

    Parameters
    ----------
    url : str
        URL pointing to an OpenQASM 3 source file.
    **kwargs
        Forwarded to :func:`parse_openqasm3_str`.

    Returns
    -------
    dict
        The parsed circuit information returned by
        :func:`parse_openqasm3_str`.
    """
    from urllib import request

    return parse_openqasm3_str(request.urlopen(url).read().decode(), **kwargs)
