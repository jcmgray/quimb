"""Tools for quantum circuit simulation using tensor networks."""

import ast
import cmath
import collections.abc
import copy
import functools
import itertools
import math
import numbers
import operator
import re
import warnings

import numpy as np
from autoray import (
    astype,
    backend_like,
    do,
    get_dtype_name,
    reshape,
)

import quimb as qu

from ..utils import (
    LRU,
    concatv,
    deprecated,
    ensure_dict,
    partition_all,
    partitionby,
    tree_map,
)
from ..utils import progbar as _progbar
from . import array_ops as ops
from .tensor_builder import (
    HTN_CP_operator_from_products,
    MPO_identity_like,
    MPS_computational_state,
    TN_from_edges_and_fill_fn,
    TN_from_sites_computational_state,
    TN_from_sites_product_state,
    gen_unique_edges,
)
from .tensor_core import (
    PTensor,
    Tensor,
    TensorNetwork,
    get_tags,
    oset_union,
    rand_uuid,
    tags_to_oset,
    tensor_contract,
)
from .tn1d.core import Dense1D, MatrixProductOperator
from .tnag.core import TensorNetworkGenOperator, TensorNetworkGenVector


def recursive_stack(x):
    if not isinstance(x, (list, tuple)):
        return x
    return do("stack", tuple(map(recursive_stack, x)))


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
        "comment": re.compile(r"^//"),
        "comment_start": re.compile(r"/\*"),
        "comment_end": re.compile(r"\*/"),
        "qreg": re.compile(r"qreg\s+(\w+)\s*\[(\d+)\];"),
        "gate": re.compile(r"(\w+)\s*(\((.+)\))?\s*(.*);"),
        "error": re.compile(r"^(if|for)"),
        "ignore": re.compile(r"^(creg|measure|barrier)"),
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
        "comment": re.compile(r"^//"),
        "comment_start": re.compile(r"/\*"),
        "comment_end": re.compile(r"\*/"),
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
        "assign": re.compile(r"(\w+)\s*=\s*(.+);"),
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
        fmt = lambda x: x if isinstance(x, str) else repr(x)
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
        from .interface import Placeholder
    except ImportError:
        return False
    return isinstance(x, Placeholder)


def _placeholder_param_vector(values):
    from .interface import Placeholder

    for value in values:
        if _is_interface_placeholder(value):
            dtype = getattr(value, "dtype", "float64")
            if dtype in (None, "unknown"):
                dtype = "float64"
            return Placeholder(np.empty((len(values),), dtype=dtype))

    raise TypeError("No placeholder values supplied.")


def parse_openqasm2_str(contents):
    """Parse the string contents of an OpenQASM 2.0 file. This parser does not
    support classical control flow is not guaranteed to check the full openqasm
    grammar.
    """
    # define regular expressions for parsing
    rgxs = get_openqasm2_regexes()

    # initialise number of qubits to zero and an empty list for gates
    sitemap = {}
    gates = []
    custom_gates = {}
    # only want to warn once about each ignored instruction
    warned = {}

    # Process each line
    in_comment = False
    lines = contents.split("\n")
    while lines:
        line = lines.pop(0).strip()
        if not line:
            # blank line
            continue
        if rgxs["comment"].match(line):
            # single comment
            continue
        if rgxs["comment_start"].match(line):
            # start of multiline comments
            in_comment = True
        if in_comment:
            # in multiline comment, check if its ending
            in_comment = not bool(rgxs["comment_end"].match(line))
            continue
        if rgxs["header"].match(line):
            # ignore standard header lines
            continue

        match = rgxs["qreg"].match(line)
        if match:
            # quantum register -> extend sites
            name, nq = match.groups()
            for i in range(int(nq)):
                sitemap[f"{name}[{i}]"] = len(sitemap)
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

            # standard gate -> add to list directly
            if params:
                params = tuple(
                    eval(param, {"pi": math.pi}) for param in params.split(",")
                )
            else:
                params = ()

            qubits = tuple(
                sitemap[qubit.strip()] for qubit in qubits.split(",")
            )
            gates.append(Gate(label, params, qubits))
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
    sitemap = {}
    registers = {}
    gates = []
    custom_gates = {}
    env = {}
    inputs = []
    symbols = {}
    expressions = {}
    warned = {}

    aliases = {
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

    def _resolve_qubit_arg(token):
        token = token.strip()
        if token in env and isinstance(env[token], (tuple, list)):
            return tuple(env[token])
        if token in registers:
            reg = registers[token]
            return reg if len(reg) > 1 else reg[0]

        match = re.fullmatch(r"(\w+)\[(.+)\]", token)
        if match:
            base, idx_expr = match.groups()
            idx = _openqasm_eval_expr(idx_expr, env)
            if not isinstance(idx, numbers.Number):
                raise NotImplementedError(
                    "Symbolic qubit indices are unsupported."
                )
            idx = int(idx)
            if base in env and isinstance(env[base], (tuple, list)):
                return env[base][idx]
            return sitemap[f"{base}[{idx}]"]

        raise NotImplementedError(f"Unknown qubit identifier: {token}")

    in_comment = False
    lines = contents.split("\n")
    while lines:
        line = lines.pop(0).strip()
        if not line:
            continue
        if rgxs["comment"].match(line):
            continue
        if rgxs["comment_start"].match(line):
            in_comment = True
        if in_comment:
            in_comment = not bool(rgxs["comment_end"].search(line))
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
                resolved = [_resolve_qubit_arg(q) for q in qubits]
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

            label = aliases.get(label.lower(), label)
            if label not in ALL_GATES:
                raise NotImplementedError(f"Unknown gate: {label}")

            raw_params = tuple(_openqasm_eval_expr(p, env) for p in params)
            resolved = [_resolve_qubit_arg(q) for q in qubits]
            sizes = {len(q) for q in resolved if isinstance(q, (tuple, list))}
            if not sizes:
                qubit_calls = [tuple(resolved)]
            else:
                if len(sizes) != 1:
                    raise NotImplementedError(
                        "Broadcasted gate args must use registers of equal "
                        "length."
                    )
                (size,) = sizes
                qubit_calls = [
                    tuple(
                        value[i] if isinstance(value, (tuple, list)) else value
                        for value in resolved
                    )
                    for i in range(size)
                ]
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


# -------------------------- core gate functions ---------------------------- #


ALL_GATES = set()
ONE_QUBIT_GATES = set()
TWO_QUBIT_GATES = set()
ALL_PARAM_GATES = set()
ONE_QUBIT_PARAM_GATES = set()
TWO_QUBIT_PARAM_GATES = set()

# the tensor tags to use for each gate (defaults to label)
GATE_TAGS = {}

# the number of qubits a gate acts on
GATE_SIZE = {}

# gates which just require a constant array
CONSTANT_GATES = {}

# gates which are parametrized
PARAM_GATES = {}

# gates which involve a non-array operation such as reindexing only
SPECIAL_GATES = {}


def register_constant_gate(name, G, num_qubits, tag=None):
    if tag is None:
        tag = name
    GATE_TAGS[name] = tag
    CONSTANT_GATES[name] = G
    GATE_SIZE[name] = num_qubits
    if num_qubits == 1:
        ONE_QUBIT_GATES.add(name)
    elif num_qubits == 2:
        TWO_QUBIT_GATES.add(name)
    ALL_GATES.add(name)


def register_param_gate(name, param_fn, num_qubits, tag=None):
    if tag is None:
        tag = name
    GATE_TAGS[name] = tag
    PARAM_GATES[name] = param_fn
    GATE_SIZE[name] = num_qubits
    if num_qubits == 1:
        ONE_QUBIT_GATES.add(name)
        ONE_QUBIT_PARAM_GATES.add(name)
    elif num_qubits == 2:
        TWO_QUBIT_GATES.add(name)
        TWO_QUBIT_PARAM_GATES.add(name)
    ALL_GATES.add(name)
    ALL_PARAM_GATES.add(name)


def register_special_gate(name, fn, num_qubits, tag=None, array=None):
    if tag is None:
        tag = name
    GATE_TAGS[name] = tag
    GATE_SIZE[name] = num_qubits
    if num_qubits == 1:
        ONE_QUBIT_GATES.add(name)
    elif num_qubits == 2:
        TWO_QUBIT_GATES.add(name)
    SPECIAL_GATES[name] = fn
    ALL_GATES.add(name)
    if array is not None:
        CONSTANT_GATES[name] = array


# constant single qubit gates
register_constant_gate("H", qu.hadamard(), 1)
register_constant_gate("X", qu.pauli("X"), 1)
register_constant_gate("Y", qu.pauli("Y"), 1)
register_constant_gate("Z", qu.pauli("Z"), 1)
register_constant_gate("S", qu.S_gate(), 1)
register_constant_gate("SDG", qu.S_gate().H, 1)
register_constant_gate("T", qu.T_gate(), 1)
register_constant_gate("TDG", qu.T_gate().H, 1)
register_constant_gate("SX", cmath.rect(1, 0.25 * math.pi) * qu.Xsqrt(), 1)
register_constant_gate(
    "SXDG", cmath.rect(1, -0.25 * math.pi) * qu.Xsqrt().H, 1
)
register_constant_gate("X_1_2", qu.Xsqrt(), 1, "X_1/2")
register_constant_gate("Y_1_2", qu.Ysqrt(), 1, "Y_1/2")
register_constant_gate("Z_1_2", qu.Zsqrt(), 1, "Z_1/2")
register_constant_gate("W_1_2", qu.Wsqrt(), 1, "W_1/2")
register_constant_gate("HZ_1_2", qu.Wsqrt(), 1, "W_1/2")


# constant two qubit gates
register_constant_gate("CX", qu.cX(), 2)
register_constant_gate("CNOT", qu.CNOT(), 2, "CX")
register_constant_gate("CY", qu.cY(), 2)
register_constant_gate("CZ", qu.cZ(), 2)
register_constant_gate("ISWAP", qu.iswap(), 2)
register_constant_gate("IS", qu.iswap(), 2, "ISWAP")


# constant three qubit gates
register_constant_gate("CCX", qu.ccX(), 3)
register_constant_gate("CCNOT", qu.ccX(), 3, "CCX")
register_constant_gate("TOFFOLI", qu.ccX(), 3, "CCX")
register_constant_gate("CCY", qu.ccY(), 3)
register_constant_gate("CCZ", qu.ccZ(), 3)
register_constant_gate("CSWAP", qu.cswap(), 3)
register_constant_gate("FREDKIN", qu.cswap(), 3, "CSWAP")


# single parametrizable gates


def rx_gate_param_gen(params):
    phi = params[0]

    with backend_like(phi):
        # get a real backend zero
        zero = phi * 0.0

        c = do("complex", do("cos", phi / 2), zero)
        s = do("complex", zero, -do("sin", phi / 2))

        return recursive_stack(((c, s), (s, c)))


register_param_gate("RX", rx_gate_param_gen, 1)


def ry_gate_param_gen(params):
    phi = params[0]

    with backend_like(phi):
        # get a real backend zero
        zero = phi * 0.0

        c = do("complex", do("cos", phi / 2), zero)
        s = do("complex", do("sin", phi / 2), zero)

        return recursive_stack(((c, -s), (s, c)))


register_param_gate("RY", ry_gate_param_gen, 1)


def rz_gate_param_gen(params):
    phi = params[0]

    with backend_like(phi):
        # get a real backend zero
        zero = phi * 0.0

        c = do("complex", do("cos", phi / 2), zero)
        s = do("complex", zero, -do("sin", phi / 2))

        # get a complex backend zero
        zero = do("complex", zero, zero)

        return recursive_stack(((c + s, zero), (zero, c - s)))


register_param_gate("RZ", rz_gate_param_gen, 1)


def u3_gate_param_gen(params):
    theta, phi, lamda = params[0], params[1], params[2]

    with backend_like(theta):
        # get a real backend zero
        zero = theta * 0.0

        theta_2 = theta / 2
        c2 = do("complex", do("cos", theta_2), zero)
        s2 = do("complex", do("sin", theta_2), zero)
        el = do("exp", do("complex", zero, lamda))
        ep = do("exp", do("complex", zero, phi))
        elp = do("exp", do("complex", zero, lamda + phi))

        return recursive_stack(((c2, -el * s2), (ep * s2, elp * c2)))


register_param_gate("U3", u3_gate_param_gen, 1)


def u2_gate_param_gen(params):
    phi, lamda = params[0], params[1]

    with backend_like(phi):
        # get a real backend zero
        zero = phi * 0.0

        c01 = -do("exp", do("complex", zero, lamda))
        c10 = do("exp", do("complex", zero, phi))
        c11 = do("exp", do("complex", zero, phi + lamda))

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        return recursive_stack(((one, c01), (c10, c11))) / 2**0.5


register_param_gate("U2", u2_gate_param_gen, 1)


def u1_gate_param_gen(params):
    lamda = params[0]

    with backend_like(lamda):
        # get a real backend zero
        zero = lamda * 0.0

        c11 = do("exp", do("complex", zero, lamda))

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        return recursive_stack(((one, zero), (zero, c11)))


register_param_gate("U1", u1_gate_param_gen, 1)
register_param_gate("PHASE", u1_gate_param_gen, 1)


# two qubit parametrizable gates


def cu3_param_gen(params):
    U3 = u3_gate_param_gen(params)

    with backend_like(U3):
        # get a 'backend zero'
        zero = 0.0 * U3[0, 0]
        # get a 'backend one'
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, one), (zero, zero))),
            (
                ((zero, zero), (U3[0, 0], U3[0, 1])),
                ((zero, zero), (U3[1, 0], U3[1, 1])),
            ),
        )

        return recursive_stack(data)


register_param_gate("CU3", cu3_param_gen, 2)


def cu2_param_gen(params):
    U2 = u2_gate_param_gen(params)

    with backend_like(U2):
        # get a 'backend zero'
        zero = 0.0 * U2[0, 0]
        # get a 'backend one'
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, one), (zero, zero))),
            (
                ((zero, zero), (U2[0, 0], U2[0, 1])),
                ((zero, zero), (U2[1, 0], U2[1, 1])),
            ),
        )

        return recursive_stack(data)


register_param_gate("CU2", cu2_param_gen, 2)


def cu1_param_gen(params):
    lamda = params[0]

    with backend_like(lamda):
        # get a real backend zero
        zero = 0.0 * lamda

        c11 = do("exp", do("complex", zero, lamda))

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, one), (zero, zero))),
            (((zero, zero), (one, zero)), ((zero, zero), (zero, c11))),
        )

        return recursive_stack(data)


register_param_gate("CU1", cu1_param_gen, 2)
register_param_gate("CPHASE", cu1_param_gen, 2)


def crx_param_gen(params):
    """Parametrized controlled X-rotation."""
    theta = params[0]

    with backend_like(theta):
        # get a real backend zero
        zero = 0.0 * theta

        ccos = do("complex", do("cos", theta / 2), zero)
        csin = do("complex", zero, -do("sin", theta / 2))

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, one), (zero, zero))),
            (((zero, zero), (ccos, csin)), ((zero, zero), (csin, ccos))),
        )

        return recursive_stack(data)


register_param_gate("CRX", crx_param_gen, 2)


def cry_param_gen(params):
    """Parametrized controlled Y-rotation."""
    theta = params[0]

    with backend_like(theta):
        # get a real backend zero
        zero = 0.0 * theta

        ccos = do("complex", do("cos", theta / 2), zero)
        csin = do("complex", do("sin", theta / 2), zero)

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, one), (zero, zero))),
            (((zero, zero), (ccos, -csin)), ((zero, zero), (csin, ccos))),
        )

        return recursive_stack(data)


register_param_gate("CRY", cry_param_gen, 2)


def crz_param_gen(params):
    """Parametrized controlled Z-rotation."""
    theta = params[0]

    with backend_like(theta):
        # get a real backend zero
        zero = 0.0 * theta

        theta_2 = theta / 2
        c = do("complex", do("cos", theta_2), zero)
        s = do("complex", zero, -do("sin", theta_2))

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, one), (zero, zero))),
            (((zero, zero), (c + s, zero)), ((zero, zero), (zero, c - s))),
        )

        return recursive_stack(data)


register_param_gate("CRZ", crz_param_gen, 2)


def fsim_param_gen(params):
    theta, phi = params[0], params[1]

    with backend_like(theta):
        # get a real backend zero
        zero = theta * 0.0

        a = do("complex", do("cos", theta), zero)
        b = do("complex", zero, -do("sin", theta))
        c = do("exp", do("complex", zero, -phi))

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, a), (b, zero))),
            (((zero, b), (a, zero)), ((zero, zero), (zero, c))),
        )

        return recursive_stack(data)


register_param_gate("FSIM", fsim_param_gen, 2)
register_param_gate("FS", fsim_param_gen, 2, "FSIM")


def fsimg_param_gen(params):
    theta, zeta, chi, gamma, phi = (
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
    )
    """Parametrized, most general number conserving two qubit gate.
    """

    with backend_like(theta):
        # get a real backend zero
        zero = 0.0 * theta

        cos = do("cos", theta)
        sin = do("sin", theta)

        c11 = do("exp", do("complex", zero, -(gamma + zeta))) * do(
            "complex", cos, zero
        )
        c12 = do("exp", do("complex", zero, -(gamma - chi))) * do(
            "complex", zero, -sin
        )
        c21 = do("exp", do("complex", zero, -(gamma + chi))) * do(
            "complex", zero, -sin
        )
        c22 = do("exp", do("complex", zero, -(gamma - zeta))) * do(
            "complex", cos, zero
        )
        c33 = do("exp", do("complex", zero, -(2 * gamma + phi)))

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, c11), (c12, zero))),
            (((zero, c21), (c22, zero)), ((zero, zero), (zero, c33))),
        )

        return recursive_stack(data)


register_param_gate("FSIMG", fsimg_param_gen, 2)


def givens_param_gen(params):
    theta = params[0]

    with backend_like(theta):
        # get a real backend zero
        zero = 0.0 * theta

        a = do("complex", do("cos", theta), zero)
        b = do("complex", do("sin", theta), zero)

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, a), (-b, zero))),
            (((zero, b), (a, zero)), ((zero, zero), (zero, one))),
        )

        return recursive_stack(data)


register_param_gate("GIVENS", givens_param_gen, num_qubits=2)


def givens2_param_gen(params):
    theta, phi = params[0], params[1]

    with backend_like(theta):
        # get a real backend zero
        zero = 0.0 * theta

        a = do("complex", do("cos", theta), zero)
        b = do("exp", do("complex", zero, phi)) * do(
            "complex", do("sin", theta), zero
        )
        b_conj = do("exp", do("complex", zero, -phi)) * do(
            "complex", do("sin", theta), zero
        )

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, a), (-b, zero))),
            (((zero, b_conj), (a, zero)), ((zero, zero), (zero, one))),
        )

        return recursive_stack(data)


register_param_gate("GIVENS2", givens2_param_gen, num_qubits=2)


def xx_plus_yy_param_gen(params):
    theta, beta = params[0], params[1]

    with backend_like(theta):
        # get a real backend zero
        zero = 0.0 * theta
        half_theta = 0.5 * theta

        a = do("complex", do("cos", half_theta), zero)
        b = do("exp", do("complex", zero, beta)) * do(
            "complex", do("sin", half_theta), zero
        )
        b_conj = do("exp", do("complex", zero, -beta)) * do(
            "complex", do("sin", half_theta), zero
        )

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((one, zero), (zero, zero)), ((zero, a), (-1j * b, zero))),
            (((zero, -1j * b_conj), (a, zero)), ((zero, zero), (zero, one))),
        )

        return recursive_stack(data)


register_param_gate("XXPLUSYY", xx_plus_yy_param_gen, num_qubits=2)


def xx_minus_yy_param_gen(params):
    theta, beta = params[0], params[1]

    with backend_like(theta):
        # get a real backend zero
        zero = 0.0 * theta
        half_theta = 0.5 * theta

        a = do("complex", do("cos", half_theta), zero)
        b = do("exp", do("complex", zero, beta)) * do(
            "complex", do("sin", half_theta), zero
        )
        b_conj = do("exp", do("complex", zero, -beta)) * do(
            "complex", do("sin", half_theta), zero
        )

        # get a complex backend zero and backend one
        zero = do("complex", zero, zero)
        one = zero + 1.0

        data = (
            (((a, zero), (zero, -1j * b_conj)), ((zero, one), (zero, zero))),
            (((zero, zero), (one, zero)), ((-1j * b, zero), (zero, a))),
        )

        return recursive_stack(data)


register_param_gate("XXMINUSYY", xx_minus_yy_param_gen, num_qubits=2)


def rxx_param_gen(params):
    r"""Parametrized two qubit XX-rotation.

    .. math::

        \mathrm{RXX}(\theta) = \exp(-i \frac{\theta}{2} X_i X_j)

    """
    theta = params[0]

    with backend_like(theta):
        # get a real 'backend zero'
        zero = 0.0 * theta

        theta_2 = theta / 2
        ccos = do("complex", do("cos", theta_2), zero)
        csin = do("complex", zero, -do("sin", theta_2))

        # get a complex backend zero
        zero = do("complex", zero, zero)

        data = (
            (((ccos, zero), (zero, csin)), ((zero, ccos), (csin, zero))),
            (((zero, csin), (ccos, zero)), ((csin, zero), (zero, ccos))),
        )

        return recursive_stack(data)


register_param_gate("RXX", rxx_param_gen, 2)


def ryy_param_gen(params):
    r"""Parametrized two qubit YY-rotation.

    .. math::

        \mathrm{RYY}(\theta) = \exp(-i \frac{\theta}{2} Y_i Y_j)

    """
    theta = params[0]

    with backend_like(theta):
        # get a real 'backend zero'
        zero = 0.0 * theta

        theta_2 = theta / 2
        ccos = do("complex", do("cos", theta_2), zero)
        csin = do("complex", zero, do("sin", theta_2))

        # get a complex backend zero
        zero = do("complex", zero, zero)

        data = (
            (((ccos, zero), (zero, csin)), ((zero, ccos), (-csin, zero))),
            (((zero, -csin), (ccos, zero)), ((csin, zero), (zero, ccos))),
        )

        return recursive_stack(data)


register_param_gate("RYY", ryy_param_gen, 2)


def rzz_param_gen(params):
    r"""Parametrized two qubit ZZ-rotation.

    .. math::

        \mathrm{RZZ}(\theta) = \exp(-i \frac{\theta}{2} Z_i Z_j)

    """
    theta = params[0]

    with backend_like(theta):
        # get a real 'backend zero'
        zero = 0.0 * theta

        theta_2 = theta / 2
        c00 = c11 = do("complex", do("cos", theta_2), do("sin", -theta_2))
        c01 = c10 = do("complex", do("cos", theta_2), do("sin", theta_2))

        # get a complex backend zero
        zero = do("complex", zero, zero)

        data = (
            (((c00, zero), (zero, zero)), ((zero, c01), (zero, zero))),
            (((zero, zero), (c10, zero)), ((zero, zero), (zero, c11))),
        )

        return recursive_stack(data)


register_param_gate("RZZ", rzz_param_gen, 2)


def su4_gate_param_gen(params):
    """See https://arxiv.org/abs/quant-ph/0308006 - Fig. 7.
    params:
    #     theta1, phi1, lamda1,
    #     theta2, phi2, lamda2,
    #     theta3, phi3, lamda3,
    #     theta4, phi4, lamda4,
    #     t1, t2, t3,
    """

    TA1 = Tensor(u3_gate_param_gen(params[0:3]), ["a1", "a0"])
    TA2 = Tensor(u3_gate_param_gen(params[3:6]), ["b1", "b0"])

    cnot = do(
        "array",
        qu.CNOT().reshape(2, 2, 2, 2),
        like=params,
        dtype=TA1.data.dtype,
    )

    TNOTC1 = Tensor(cnot, ["b2", "a2", "b1", "a1"])
    TRz1 = Tensor(rz_gate_param_gen(params[12:13]), inds=["a3", "a2"])
    TRy2 = Tensor(ry_gate_param_gen(params[13:14]), inds=["b3", "b2"])
    TCNOT2 = Tensor(cnot, ["a5", "b4", "a3", "b3"])
    TRy3 = Tensor(ry_gate_param_gen(params[14:15]), inds=["b5", "b4"])
    TNOTC3 = Tensor(cnot, ["b6", "a6", "b5", "a5"])
    TA3 = Tensor(u3_gate_param_gen(params[6:9]), ["a7", "a6"])
    TA4 = Tensor(u3_gate_param_gen(params[9:12]), ["b7", "b6"])

    return tensor_contract(
        TA1,
        TA2,
        TNOTC1,
        TRz1,
        TRy2,
        TCNOT2,
        TRy3,
        TNOTC3,
        TA3,
        TA4,
        output_inds=["a7", "b7"] + ["a0", "b0"],
        optimize="auto-hq",
    ).data


register_param_gate("SU4", su4_gate_param_gen, 2)


# special non-tensor gates

_MPS_METHODS = {
    "auto-mps",
    "nonlocal",
    "swap+split",
}


def apply_swap(psi, i, j, **gate_opts):
    contract = gate_opts.pop("contract", None)

    if contract not in _MPS_METHODS:
        # just do swap by lazily reindexing
        iind, jind = map(psi.site_ind, (int(i), int(j)))
        psi.reindex_({iind: jind, jind: iind})

    else:
        # tensors are absorbed so propagate_tags is not needed
        gate_opts.pop("propagate_tags", None)

        if contract == "nonlocal":
            psi.gate_nonlocal_(qu.swap(2), (i, j), **gate_opts)
        else:  # {"swap+split", "auto-mps"}:
            psi.swap_sites_with_compress_(i, j, **gate_opts)


register_special_gate("SWAP", apply_swap, 2, array=qu.swap(2))
register_special_gate("IDEN", lambda *_, **__: None, 1, array=qu.identity(2))


def build_controlled_gate_htn(
    ncontrol,
    gate,
    upper_inds,
    lower_inds,
    tags_each=None,
    tags_all=None,
    bond_ind=None,
):
    """Build a low rank hyper tensor network (CP-decomp like) representation of
    a multi controlled gate.
    """
    ngate = len(gate.qubits)
    gate_shape = (2,) * (2 * ngate)
    array = gate.array.reshape(gate_shape)

    I2 = qu.identity(2, dtype=array.dtype)
    IG = qu.identity(2**ngate, dtype=array.dtype).reshape(gate_shape)
    p1 = qu.down(qtype="dop", dtype=array.dtype)  # |1><1|

    array_seqs = [[I2] * ncontrol + [IG], [p1] * ncontrol + [array - IG]]

    # might need to group indices and tags on the target gate if multi-qubit
    if ngate > 1:
        upper_inds = (*upper_inds[:ncontrol], upper_inds[ncontrol:])
        lower_inds = (*lower_inds[:ncontrol], lower_inds[ncontrol:])
        tags_each = (*tags_each[:ncontrol], tags_each[ncontrol:])

    htn = HTN_CP_operator_from_products(
        array_seqs,
        upper_inds=upper_inds,
        lower_inds=lower_inds,
        tags_each=tags_each,
        tags_all=tags_all,
        bond_ind=bond_ind,
    )

    return htn


def _apply_controlled_gate_mps(psi, gate, tags=None, **gate_opts):
    """Apply a multi-controlled gate to a state represented as an MPS."""
    submpo = gate.build_mpo()
    where = sorted((*gate.controls, *gate.qubits))
    psi.gate_with_submpo_(submpo, where, **gate_opts)


def _apply_controlled_gate_htn(
    psi, gate, tags=None, propagate_tags="register", **gate_opts
):
    assert propagate_tags == "register"

    all_qubits = (*gate.controls, *gate.qubits)
    ncontrol = len(gate.controls)
    ngate = len(gate.qubits)
    ntotal = ncontrol + ngate

    upper_inds = [rand_uuid() for _ in range(ntotal)]
    lower_inds = [rand_uuid() for _ in range(ntotal)]
    tags_sequence = [psi.site_tag(i) for i in all_qubits]

    htn = build_controlled_gate_htn(
        ncontrol,
        gate,
        upper_inds=upper_inds,
        lower_inds=lower_inds,
        tags_each=tags_sequence,
        tags_all=tags,
    )

    psi.gate_inds_with_tn_(
        [psi.site_ind(i) for i in all_qubits],
        htn,
        lower_inds,
        upper_inds,
        **gate_opts,
    )


def apply_controlled_gate(
    psi,
    gate,
    tags=None,
    contract="auto-split-gate",
    propagate_tags="register",
    **gate_opts,
):
    if contract in ("auto-mps", "nonlocal"):
        _apply_controlled_gate_mps(psi, gate, tags=tags, **gate_opts)
    elif contract in (
        "auto-split-gate",
        "split-gate",
    ):
        _apply_controlled_gate_htn(
            psi, gate, tags=tags, propagate_tags=propagate_tags, **gate_opts
        )
    else:
        raise ValueError(
            f"Contract method '{contract}' not "
            "supported for multi-controlled gates."
        )


@functools.lru_cache(2**15)
def _cached_param_gate_build(fn, params):
    return fn(params)


class Gate:
    """A simple class for storing the details of a quantum circuit gate.

    Parameters
    ----------
    label : str
        The name or 'identifier' of the gate.
    params : Iterable[float]
        The parameters of the gate.
    qubits : Iterable[int], optional
        Which qubits the gate acts on.
    controls : Iterable[int], optional
        Which qubits are the controls.
    round : int, optional
        If given, which round or layer the gate is part of.
    parametrize : bool, optional
        Whether the gate will correspond to a parametrized tensor.
    """

    __slots__ = (
        "_label",
        "_params",
        "_qubits",
        "_controls",
        "_round",
        "_parametrize",
        "_tag",
        "_special",
        "_constant",
        "_array",
    )

    def __init__(
        self,
        label,
        params,
        qubits=None,
        controls=None,
        round=None,
        parametrize=False,
    ):
        self._label = label.upper()

        if self._label not in ALL_GATES:
            raise ValueError(f"Unknown gate: {self._label}.")

        self._params = ops.asarray(params)
        if qubits is None:
            self._qubits = None
        else:
            self._qubits = tuple(qubits)

        if controls is None:
            self._controls = None
        else:
            self._controls = tuple(controls)

        self._round = int(round) if round is not None else round
        self._parametrize = bool(parametrize)

        self._tag = GATE_TAGS[self._label]
        self._special = self._label in SPECIAL_GATES
        self._constant = self._label in CONSTANT_GATES
        if (self._special or self._constant) and self._parametrize:
            raise ValueError(f"Cannot parametrize the gate: {self._label}.")
        self._array = None

    @classmethod
    def from_raw(cls, U, qubits=None, controls=None, round=None):
        new = object.__new__(cls)
        new._label = f"RAW{id(U)}"
        new._params = "raw"
        if qubits is None:
            new._qubits = None
        else:
            new._qubits = tuple(qubits)
        if controls is None:
            new._controls = None
        else:
            new._controls = tuple(controls)
        new._round = int(round) if round is not None else round
        new._special = False
        new._parametrize = isinstance(U, ops.PArray)
        new._tag = None
        new._array = U
        return new

    def copy(self):
        new = object.__new__(self.__class__)
        new._label = self._label
        new._params = self._params
        new._qubits = self._qubits
        new._controls = self._controls
        new._round = self._round
        new._parametrize = self._parametrize
        new._tag = self._tag
        new._special = self._special
        new._constant = self._constant
        new._array = self._array
        return new

    @property
    def label(self):
        return self._label

    @property
    def params(self):
        return self._params

    @property
    def qubits(self):
        return self._qubits

    @qubits.setter
    def qubits(self, qubits):
        if qubits is None:
            self._qubits = None
        else:
            self._qubits = tuple(qubits)

    @property
    def total_qubit_count(self):
        nq = len(self._qubits)
        if self._controls:
            nq += len(self._controls)
        return nq

    @property
    def controls(self):
        return self._controls

    @property
    def round(self):
        return self._round

    @property
    def special(self):
        return self._special

    @property
    def parametrize(self):
        return self._parametrize

    @property
    def tag(self):
        return self._tag

    def copy_with(self, **kwargs):
        """Take a copy of this gate but with some attributes changed."""
        label = kwargs.get("label", self._label)
        params = kwargs.get("params", self._params)
        qubits = kwargs.get("qubits", self._qubits)
        controls = kwargs.get("controls", self._controls)
        round = kwargs.get("round", self._round)
        parametrize = kwargs.get("parametrize", self._parametrize)

        if isinstance(params, str) and (params == "raw"):
            return self.from_raw(
                U=self._array,
                qubits=qubits,
                controls=controls,
                round=round,
            )
        else:
            return self.__class__(
                label=label,
                params=params,
                qubits=qubits,
                controls=controls,
                round=round,
                parametrize=parametrize,
            )

    def build_array(self):
        """Build the array representation of the gate. For controlled gates
        this *excludes* the control qubits.
        """
        if self._special and (self._label not in CONSTANT_GATES):
            # these don't have an array representation
            raise ValueError(f"{self.label} gates have no array to build.")

        if self._constant:
            # simply return the constant array
            return CONSTANT_GATES[self._label]

        # build the array
        param_fn = PARAM_GATES[self._label]
        if self._parametrize:
            # either lazily, as tensor will be parametrized
            shape = (2,) * (2 * len(self._qubits))
            return ops.PArray(param_fn, self._params, shape=shape)

        # or cached directly into array
        try:
            return _cached_param_gate_build(param_fn, self._params)
        except TypeError:
            return param_fn(self._params)

    @property
    def array(self):
        if self._array is None:
            self._array = self.build_array()
        return self._array

    def build_mpo(self, L=None, **kwargs):
        """Build an MPO representation of this gate."""
        G = self.array

        if L is None:
            L = max((*self.qubits, *self.controls), default=0) + 1

        if not self.controls:
            return MatrixProductOperator.from_dense(
                G, sites=self.qubits, L=L, **kwargs
            )

        IG = qu.identity(2 ** len(self.qubits))
        IG = reshape(IG, G.shape)
        p1 = qu.down(qtype="dop")

        # form (G - 1) on target qubits
        mpo = MatrixProductOperator.from_dense(
            G - IG, sites=self.qubits, L=L, **kwargs
        )

        # take tensor product with |11...><11...| on controls
        mpo.fill_empty_sites_(mode=self.controls, fill_array=p1)

        # add with identity on all qubits
        mpo_I = MPO_identity_like(
            mpo, sites=sorted((*self.qubits, *self.controls))
        )

        return mpo.add_MPO_(mpo_I)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}("
            + f"label={self._label}, "
            + f"params={self._params}, "
            + f"qubits={self._qubits}"
            + (f", controls={self._controls})" if self._controls else "")
            + (f", round={self._round}" if self._round is not None else "")
            + (
                f", parametrize={self._parametrize})"
                if self._parametrize
                else ""
            )
            + ")>"
        )


def sample_bitstring_from_prob_ndarray(p, seed=None):
    """Sample a bitstring from n-dimensional tensor ``p`` of probabilities.

    Examples
    --------

        >>> import numpy as np
        >>> p = np.zeros(shape=(2, 2, 2, 2, 2))
        >>> p[0, 1, 0, 1, 1] = 1.0
        >>> sample_bitstring_from_prob_ndarray(p)
        '01011'
    """
    rng = np.random.default_rng(seed)
    b = rng.choice(p.size, p=p.ravel())
    return f"{b:0>{p.ndim}b}"


def rehearsal_dict(tn, tree):
    return {
        "tn": tn,
        "tree": tree,
        "W": tree.contraction_width(),
        "C": math.log10(max(tree.contraction_cost(), 1)),
    }


def parse_to_gate(
    gate_id,
    *gate_args,
    params=None,
    qubits=None,
    controls=None,
    gate_round=None,
    parametrize=None,
):
    """Map all types of gate specification into a `Gate` object."""

    if isinstance(gate_id, Gate):
        # already a gate
        if gate_args:
            raise ValueError(
                "You cannot specify ``gate_args`` for an already "
                "encapsulated `Gate` object."
            )

        if any((params, qubits, controls, gate_round, parametrize)):
            raise ValueError(
                "You cannot specify ``controls`` or ``gate_round`` for an "
                "already encapsulated gate - supply directly to the  `Gate` "
                "constructor instead."
            )
        return gate_id

    if isinstance(gate_id, tuple):
        # if given a tuple just unpack it
        if gate_args:
            raise ValueError(
                "You cannot specify ``gate_args`` when supplying a tuple."
            )
        gate_id, gate_args = gate_id[0], gate_id[1:]

    if hasattr(gate_id, "shape") and not isinstance(gate_id, str):
        # raw gate (numpy strings have a shape - ignore those)

        if parametrize is not None:
            raise ValueError(
                "You cannot specify ``parametrize`` for raw gate, supply a "
                "``PArray`` instead."
            )

        return Gate.from_raw(
            U=gate_id,
            qubits=gate_args,
            controls=controls,
            round=gate_round,
        )

    # else gate is specified as a tuple or kwargs

    if isinstance(gate_id, numbers.Integral) or gate_id.isdigit():
        # gate round given as first entry of tuple
        if gate_round is None:
            # explicilty specified ``gate_round`` takes precedence
            gate_round = gate_id
        gate_id, gate_args = gate_args[0], gate_args[1:]

    if parametrize is None:
        parametrize = False

    if gate_args:
        if any((params, qubits)):
            raise ValueError(
                "You cannot specify ``params`` or ``qubits`` "
                "when supplying ``gate_args``."
            )

        nq = GATE_SIZE[gate_id.upper()]
        (
            params,
            qubits,
        ) = (
            gate_args[:-nq],
            gate_args[-nq:],
        )

    else:
        # qubits and params specified directly
        if params is None:
            params = ()

    return Gate(
        label=gate_id,
        params=params,
        qubits=qubits,
        controls=controls,
        round=gate_round,
        parametrize=parametrize,
    )


# --------------------------- main circuit class ---------------------------- #


class Circuit:
    """Class for simulating quantum circuits using tensor networks. The class
    keeps a list of :class:`Gate` objects in sync with a tensor network
    representing the current state of the circuit.

    Parameters
    ----------
    N : int, optional
        The number of qubits.
    psi0 : TensorNetwork1DVector, optional
        The initial state, assumed to be ``|00000....0>`` if not given. The
        state is always copied and the tag ``PSI0`` added.
    gate_opts : dict_like, optional
        Default keyword arguments to supply to each
        :func:`~quimb.tensor.tn1d.core.gate_TN_1D` call during the circuit.
    gate_contract : str, optional
        Shortcut for setting the default `'contract'` option in `gate_opts`.
    gate_propagate_tags : str, optional
        Shortcut for setting the default `'propagate_tags'` option in
        `gate_opts`.
    tags : str or sequence of str, optional
        Tag(s) to add to the initial wavefunction tensors (whether these are
        propagated to the rest of the circuit's tensors depends on
        ``gate_opts``).
    psi0_dtype : str, optional
        Ensure the initial state has this dtype.
    psi0_tag : str, optional
        Ensure the initial state has this tag.
    tag_gate_numbers : bool, optional
        Whether to tag each gate tensor with its number in the circuit, like
        ``"GATE_{g}"``. This is required for updating the circuit parameters.
    gate_tag_id : str, optional
        The format string for tagging each gate tensor, by default e.g.
        ``"GATE_{g}"``.
    tag_gate_rounds : bool, optional
        Whether to tag each gate tensor with its number in the circuit, like
        ``"ROUND_{r}"``.
    round_tag_id : str, optional
        The format string for tagging each round of gates, by default e.g.
        ``"ROUND_{r}"``.
    tag_gate_labels : bool, optional
        Whether to tag each gate tensor with its gate type label, e.g.
        ``{"X_1/2", "ISWAP", "CCX", ...}``..
    bra_site_ind_id : str, optional
        Use this to label 'bra' site indices when creating certain (mostly
        internal) intermediate tensor networks.
    dtype : str, optional
        A default dtype to perform calculations in. Depending on
        `convert_eager`, this is enforced *after* circuit construction
        and simplification (the default for exact simulation), or eagerly to
        the initial state and as gates are applied (the default for MPS
        simulation).
    to_backend : callable, optional
        If given, apply this function to both the initial state arrays and to
        every gate as it is applied.
    convert_eager : bool, optional
        Whether to eagerly perform dtype casting and application of
        `to_backend` as gates are supplied, or wait until after the necessary
        TNs for a particular task such as sampling are formed and simplified.
        Deferred conversion (`convert_eager=False`) is the default mode for
        full contraction.

    Attributes
    ----------
    psi : TensorNetwork1DVector
        The current circuit wavefunction as a tensor network.
    uni : TensorNetwork1DOperator
        The current circuit unitary operator as a tensor network.
    gates : tuple[Gate]
        The gates in the circuit.

    Examples
    --------

    Create 3-qubit GHZ-state:

        >>> qc = qtn.Circuit(3)
        >>> gates = [
                ('H', 0),
                ('H', 1),
                ('CNOT', 1, 2),
                ('CNOT', 0, 2),
                ('H', 0),
                ('H', 1),
                ('H', 2),
            ]
        >>> qc.apply_gates(gates)
        >>> qc.psi
        <TensorNetwork1DVector(tensors=12, indices=14, L=3, max_bond=2)>

        >>> qc.psi.to_dense().round(4)
        qarray([[ 0.7071+0.j],
                [ 0.    +0.j],
                [ 0.    +0.j],
                [-0.    +0.j],
                [-0.    +0.j],
                [ 0.    +0.j],
                [ 0.    +0.j],
                [ 0.7071+0.j]])

        >>> for b in qc.sample(10):
        ...     print(b)
        000
        000
        111
        000
        111
        111
        000
        111
        000
        000

    See Also
    --------
    Gate
    """

    def __init__(
        self,
        N=None,
        psi0=None,
        gate_opts=None,
        gate_contract="auto-split-gate",
        gate_propagate_tags="register",
        tags=None,
        psi0_dtype="complex128",
        psi0_tag="PSI0",
        tag_gate_numbers=True,
        gate_tag_id="GATE_{}",
        tag_gate_rounds=True,
        round_tag_id="ROUND_{}",
        tag_gate_labels=True,
        bra_site_ind_id="b{}",
        dtype=None,
        to_backend=None,
        convert_eager=False,
    ):
        if (N is None) and (psi0 is None):
            raise ValueError("You must supply one of `N` or `psi0`.")

        elif psi0 is None:
            self.N = N
            self._psi = self._init_state(N, dtype=psi0_dtype)

        elif N is None:
            self._psi = psi0.copy()
            self.N = psi0.nsites

        else:
            if N != psi0.nsites:
                raise ValueError("`N` doesn't match `psi0`.")
            self.N = N
            self._psi = psi0.copy()

        self._psi.add_tag(psi0_tag)

        if tags is not None:
            if isinstance(tags, str):
                tags = (tags,)
            for tag in tags:
                self._psi.add_tag(tag)

        self.tag_gate_numbers = tag_gate_numbers
        self.tag_gate_rounds = tag_gate_rounds
        self.tag_gate_labels = tag_gate_labels

        self.dtype = dtype
        self.to_backend = to_backend
        self.convert_eager = convert_eager
        if self.convert_eager:
            self._maybe_convert(self._psi)
        self._backend_gate_cache = {}

        self.gate_opts = ensure_dict(gate_opts)
        self.gate_opts.setdefault("contract", gate_contract)
        self.gate_opts.setdefault("propagate_tags", gate_propagate_tags)
        self._gates = []

        self._ket_site_ind_id = self._psi.site_ind_id
        self._bra_site_ind_id = bra_site_ind_id
        self._gate_tag_id = gate_tag_id
        self._round_tag_id = round_tag_id

        if self._ket_site_ind_id == self._bra_site_ind_id:
            raise ValueError(
                "The 'ket' and 'bra' site ind ids clash : '{}' and '{}".format(
                    self._ket_site_ind_id, self._bra_site_ind_id
                )
            )

        self._sample_n_gates = -1
        self._storage = dict()
        self._sampled_conditionals = dict()
        self._named_params = {}
        self._named_param_exprs = {}

    def copy(self):
        """Copy the circuit and its state."""
        new = object.__new__(self.__class__)
        new.N = self.N
        new._psi = self._psi.copy()
        new.gate_opts = tree_map(lambda x: x, self.gate_opts)
        new.tag_gate_numbers = self.tag_gate_numbers
        new.tag_gate_rounds = self.tag_gate_rounds
        new.tag_gate_labels = self.tag_gate_labels
        new.to_backend = self.to_backend
        new.dtype = self.dtype
        new.convert_eager = self.convert_eager
        new._backend_gate_cache = self._backend_gate_cache
        new._gates = self._gates.copy()
        new._ket_site_ind_id = self._ket_site_ind_id
        new._bra_site_ind_id = self._bra_site_ind_id
        new._gate_tag_id = self._gate_tag_id
        new._round_tag_id = self._round_tag_id
        new._sample_n_gates = self._sample_n_gates
        new._storage = self._storage.copy()
        new._sampled_conditionals = self._sampled_conditionals.copy()
        new._named_params = copy.copy(self._named_params)
        new._named_param_exprs = copy.copy(self._named_param_exprs)
        return new

    def _maybe_convert(self, obj, dtype=None):
        istn = isinstance(obj, TensorNetwork)

        if dtype is None:
            # use default dtype
            dtype = self.dtype

        if dtype is not None:
            # cast array or TN to dtype
            if istn:
                obj.astype_(dtype)
            else:
                if get_dtype_name(obj) != dtype:
                    obj = astype(obj, dtype)

        if self.to_backend is not None:
            # once dtype is enforced, apply to_backend
            # for e.g. gpu transfer etc
            if istn:
                obj.apply_to_arrays(self.to_backend)
            else:
                obj = self.to_backend(obj)

        return obj

    def apply_to_arrays(self, fn):
        """Apply a function to all the arrays in the circuit."""
        self._psi.apply_to_arrays(fn)
        self._named_params = tree_map(fn, self._named_params)

    @staticmethod
    def _normalize_named_param_value(value):
        if _is_interface_placeholder(value):
            return value
        if isinstance(value, numbers.Number):
            return ops.asarray(value)
        return value

    @property
    def named_params(self):
        """Named circuit parameters and their current values."""
        return copy.copy(self._named_params)

    @property
    def named_param_names(self):
        """Names of registered circuit parameters."""
        return tuple(self._named_params)

    @property
    def param_expressions(self):
        """Gate parameter expressions keyed by gate index."""
        return copy.copy(self._named_param_exprs)

    def register_named_params(self, named_params, gate_expressions=None):
        """Register named circuit parameters and gate dependencies.

        Parameters
        ----------
        named_params : sequence[str] or mapping[str, scalar]
            Either names to register, which default to ``nan`` until bound,
            or a mapping supplying initial values.
        gate_expressions : mapping[int, tuple], optional
            Mapping from gate index to the expressions used to generate that
            gate's parameters. Each expression can be a constant, a string
            expression referencing the named parameters, or a callable taking
            the current named parameter mapping.
        """
        if isinstance(named_params, collections.abc.Mapping):
            self._named_params = {
                name: self._normalize_named_param_value(value)
                for name, value in named_params.items()
            }
        else:
            self._named_params = {
                name: self._normalize_named_param_value(float("nan"))
                for name in tuple(named_params)
            }

        if gate_expressions is None:
            gate_expressions = {}

        normalized_gate_expressions = {}
        for i, exprs in gate_expressions.items():
            i = int(i)
            exprs = tuple(exprs)

            if not (0 <= int(i) < len(self._gates)):
                raise ValueError(
                    "Named parameter expressions reference unknown gate "
                    f"index: {i}"
                )

            gate = self._gates[i]
            if not gate.parametrize:
                raise ValueError(
                    "Named parameter expressions require parametrized gate "
                    f"indices, got non-parametrized gate: {i}"
                )

            if len(exprs) != len(gate.params):
                raise ValueError(
                    "Named parameter expression arity does not match gate "
                    f"{i}: expected {len(gate.params)}, got {len(exprs)}"
                )

            normalized_gate_expressions[i] = exprs

        self._named_param_exprs = normalized_gate_expressions
        self._apply_named_param_updates()
        self.clear_storage()

    def _set_gate_params(self, i, params):
        self._psi[self.gate_tag(i)].params = params
        self._gates[i] = self._gates[i].copy_with(params=ops.asarray(params))

    def _apply_named_param_updates(self):
        if not self._named_param_exprs:
            return

        env = dict(self._named_params)
        for i, exprs in self._named_param_exprs.items():
            values = tuple(_openqasm_eval_expr(expr, env) for expr in exprs)
            if any(isinstance(x, str) for x in values):
                raise ValueError(
                    "Named parameter binding left unresolved symbolic values "
                    f"for gate {i}: {values!r}"
                )
            if any(_is_interface_placeholder(x) for x in values):
                values = _placeholder_param_vector(values)
            self._set_gate_params(i, values)

    def get_params(self):
        """Get a pytree - in this case a dict - of all the parameters in the
        circuit.

        Returns
        -------
        dict
            Dictionary containing any named parameters plus any directly
            parametrized gates not driven by named parameter expressions.
        """
        params = dict(self._named_params)
        managed_gates = set(self._named_param_exprs)
        params.update(
            {
                i: self._psi[self.gate_tag(i)].params
                for i, gate in enumerate(self._gates)
                if gate.parametrize and i not in managed_gates
            }
        )
        return params

    def set_params(self, params):
        """Set the parameters of the circuit.

        Parameters
        ----------
        params : dict
            Dictionary mapping gate numbers and/or registered named parameter
            names to new values.
        """
        if params is None:
            params = {}

        named_updates = {k: v for k, v in params.items() if isinstance(k, str)}
        gate_updates = {
            k: v for k, v in params.items() if not isinstance(k, str)
        }

        if named_updates and not self._named_params:
            raise TypeError(
                "String-keyed parameters require registered named parameters."
            )

        extra = set(named_updates) - set(self._named_params)
        if extra:
            raise ValueError(
                "Unknown named parameter values supplied for: "
                + ", ".join(sorted(extra))
            )

        overlap = set(gate_updates) & set(self._named_param_exprs)
        if overlap:
            raise ValueError(
                "Cannot directly set gate parameters managed by named "
                "parameter expressions: "
                + ", ".join(map(str, sorted(overlap)))
            )

        if named_updates:
            self._named_params.update(
                {
                    name: self._normalize_named_param_value(value)
                    for name, value in named_updates.items()
                }
            )
            self._apply_named_param_updates()

        for i, p in gate_updates.items():
            self._set_gate_params(i, p)
        self.clear_storage()

    @classmethod
    def from_qsim_str(cls, contents, progbar=False, **circuit_opts):
        """Generate a ``Circuit`` instance from a 'qsim' string."""
        info = parse_qsim_str(contents)
        qc = cls(info["n"], **circuit_opts)
        qc.apply_gates(info["gates"], progbar=progbar)
        return qc

    @classmethod
    def from_qsim_file(cls, fname, progbar=False, **circuit_opts):
        """Generate a ``Circuit`` instance from a 'qsim' file.

        The qsim file format is described here:
        https://quantumai.google/qsim/input_format.
        """
        info = parse_qsim_file(fname)
        qc = cls(info["n"], **circuit_opts)
        qc.apply_gates(info["gates"], progbar=progbar)
        return qc

    @classmethod
    def from_qsim_url(cls, url, progbar=False, **circuit_opts):
        """Generate a ``Circuit`` instance from a 'qsim' url."""
        info = parse_qsim_url(url)
        qc = cls(info["n"], **circuit_opts)
        qc.apply_gates(info["gates"], progbar=progbar)
        return qc

    from_qasm = deprecated(from_qsim_str, "from_qasm", "from_qsim_str")
    from_qasm_file = deprecated(
        from_qsim_file, "from_qasm_file", "from_qsim_file"
    )
    from_qasm_url = deprecated(from_qsim_url, "from_qasm_url", "from_qsim_url")

    @classmethod
    def from_openqasm2_str(cls, contents, progbar=False, **circuit_opts):
        """Generate a ``Circuit`` instance from an OpenQASM 2.0 string."""
        info = parse_openqasm2_str(contents)
        qc = cls(info["n"], **circuit_opts)
        qc.apply_gates(info["gates"], progbar)
        return qc

    @classmethod
    def from_openqasm2_file(cls, fname, progbar=False, **circuit_opts):
        """Generate a ``Circuit`` instance from an OpenQASM 2.0 file."""
        info = parse_openqasm2_file(fname)
        qc = cls(info["n"], **circuit_opts)
        qc.apply_gates(info["gates"], progbar=progbar)
        return qc

    @classmethod
    def from_openqasm2_url(cls, url, progbar=False, **circuit_opts):
        """Generate a ``Circuit`` instance from an OpenQASM 2.0 url."""
        info = parse_openqasm2_url(url)
        qc = cls(info["n"], **circuit_opts)
        qc.apply_gates(info["gates"], progbar=progbar)
        return qc

    @classmethod
    def from_openqasm3_str(cls, contents, progbar=False, **circuit_opts):
        """Construct a circuit from an OpenQASM 3.0 string.

        Parameters
        ----------
        contents : str
            The OpenQASM 3 source code to parse.
        progbar : bool, optional
            Whether to show a progress bar while applying the parsed gates.
        **circuit_opts
            Options forwarded to the ``Circuit`` constructor.

        Returns
        -------
        Circuit
            A circuit populated with the parsed gates. If symbolic ``input``
            declarations are present, they are registered as generic named
            circuit parameters so that :meth:`set_params` can bind them later.
        """
        info = parse_openqasm3_str(contents)
        qc = cls(info["n"], **circuit_opts)
        qc.apply_gates(info["gates"], progbar=progbar)
        qc.register_named_params(
            {
                name: (value if not isinstance(value, str) else float("nan"))
                for name, value in info["symbols"].items()
            },
            info["expressions"],
        )
        return qc

    @classmethod
    def from_openqasm3_file(cls, fname, progbar=False, **circuit_opts):
        """Construct a circuit from an OpenQASM 3.0 file.

        Parameters
        ----------
        fname : str or path-like
            Path to the OpenQASM 3 file.
        progbar : bool, optional
            Whether to show a progress bar while applying the parsed gates.
        **circuit_opts
            Options forwarded to the ``Circuit`` constructor.

        Returns
        -------
        Circuit
            The parsed circuit instance.
        """
        with open(fname) as f:
            return cls.from_openqasm3_str(
                f.read(), progbar=progbar, **circuit_opts
            )

    @classmethod
    def from_openqasm3_url(cls, url, progbar=False, **circuit_opts):
        """Construct a circuit from an OpenQASM 3.0 URL.

        Parameters
        ----------
        url : str
            URL pointing to an OpenQASM 3 source file.
        progbar : bool, optional
            Whether to show a progress bar while applying the parsed gates.
        **circuit_opts
            Options forwarded to the ``Circuit`` constructor.

        Returns
        -------
        Circuit
            The parsed circuit instance.
        """
        from urllib import request

        return cls.from_openqasm3_str(
            request.urlopen(url).read().decode(),
            progbar=progbar,
            **circuit_opts,
        )

    @classmethod
    def from_gates(cls, gates, N=None, progbar=False, **kwargs):
        """Generate a ``Circuit`` instance from a sequence of gates.

        Parameters
        ----------
        gates : sequence[Gate] or sequence[tuple]
            The sequence of gates to apply.
        N : int, optional
            The number of qubits. If not given, will be inferred from the
            gates.
        progbar : bool, optional
            Whether to show a progress bar.
        kwargs
            Supplied to the ``Circuit`` constructor.
        """
        if N is None:
            gates = tuple(gates)

            N = 0
            for gate in gates:
                gate = parse_to_gate(gate)
                if gate.qubits:
                    N = max(N, max(gate.qubits) + 1)
                if gate.controls:
                    N = max(N, max(gate.controls) + 1)

        qc = cls(N, **kwargs)
        qc.apply_gates(gates, progbar=progbar)
        return qc

    @property
    def gates(self):
        return tuple(self._gates)

    @property
    def num_gates(self):
        return len(self._gates)

    def ket_site_ind(self, i):
        """Get the site index for the given qubit."""
        return self._ket_site_ind_id.format(i)

    def bra_site_ind(self, i):
        """Get the 'bra' site index for the given qubit, if forming an operator."""
        return self._bra_site_ind_id.format(i)

    def gate_tag(self, g):
        """Get the tag for the given gate, indexed linearly."""
        return self._gate_tag_id.format(g)

    def round_tag(self, r):
        """Get the tag for the given round (/layer)."""
        return self._round_tag_id.format(r)

    def _init_state(self, N, dtype="complex128"):
        return TN_from_sites_computational_state(
            site_map={i: "0" for i in range(N)}, dtype=dtype
        )

    def _apply_gate(self, gate, tags=None, **gate_opts):
        """Apply a ``Gate`` to this ``Circuit``. This is the main method that
        all calls to apply a gate should go through.

        Parameters
        ----------
        gate : Gate
            The gate to apply.
        tags : str or sequence of str, optional
            Tags to add to the gate tensor(s).
        """
        tags = tags_to_oset(tags)
        if self.tag_gate_numbers:
            tags.add(self.gate_tag(self.num_gates))
        if self.tag_gate_rounds and (gate.round is not None):
            tags.add(self.round_tag(gate.round))
        if self.tag_gate_labels and (gate.tag is not None):
            tags.add(gate.tag)

        # overide any default gate opts
        opts = {**self.gate_opts, **gate_opts}

        if gate.controls:
            # handle extra (low-rank) control structure
            apply_controlled_gate(self._psi, gate, tags=tags, **opts)

        elif gate.special:
            # these are specified as a general function
            SPECIAL_GATES[gate.label](
                self._psi, *gate.params, *gate.qubits, **opts
            )

        else:
            # gate supplied as a matrix/tensor
            G = gate.array

            if self.convert_eager:
                key = id(G)
                if key not in self._backend_gate_cache:
                    self._backend_gate_cache[key] = self._maybe_convert(G)
                G = self._backend_gate_cache[key]

            # apply the gate to the TN!
            self._psi.gate_(G, gate.qubits, tags=tags, **opts)

        # keep track of the gates applied
        self._gates.append(gate)

    def apply_gate(
        self,
        gate_id,
        *gate_args,
        params=None,
        qubits=None,
        controls=None,
        gate_round=None,
        parametrize=None,
        **gate_opts,
    ):
        """Apply a single gate to this tensor network quantum circuit. If
        ``gate_round`` is supplied the tensor(s) added will be tagged with
        ``'ROUND_{gate_round}'``. Alternatively, putting an integer first like
        so::

            circuit.apply_gate(10, 'H', 7)

        Is automatically translated to::

            circuit.apply_gate('H', 7, gate_round=10)

        Parameters
        ----------
        gate_id : Gate, str, or array_like
            Which gate to apply. This can be:

                - A ``Gate`` instance, i.e. with parameters and qubits already
                  specified.
                - A string, e.g. ``'H'``, ``'U3'``, etc. in which case
                  ``gate_args`` should be supplied with ``(*params, *qubits)``.
                - A raw array, in which case ``gate_args`` should be supplied
                  with ``(*qubits,)``.

        gate_args : list[str]
            The arguments to supply to it.
        gate_round : int, optional
            The gate round. If ``gate_id`` is integer-like, will also be taken
            from here, with then ``gate_id, gate_args = gate_args[0],
            gate_args[1:]``.
        gate_opts
            Supplied to the gate function, options here will override the
            default ``gate_opts``.
        """
        gate = parse_to_gate(
            gate_id,
            *gate_args,
            params=params,
            qubits=qubits,
            controls=controls,
            gate_round=gate_round,
            parametrize=parametrize,
        )
        self._apply_gate(gate, **gate_opts)

    def apply_gate_raw(
        self, U, where, controls=None, gate_round=None, **gate_opts
    ):
        """Apply the raw array ``U`` as a gate on qubits in ``where``. It will
        be assumed to be unitary for the sake of computing reverse lightcones.
        """
        gate = Gate.from_raw(U, where, controls=controls, round=gate_round)
        self._apply_gate(gate, **gate_opts)

    def apply_gates(self, gates, progbar=False, **gate_opts):
        """Apply a sequence of gates to this tensor network quantum circuit.

        Parameters
        ----------
        gates : Sequence[Gate] or Sequence[Tuple]
            The sequence of gates to apply.
        gate_opts
            Supplied to :meth:`~quimb.tensor.circuit.Circuit.apply_gate`.
        """
        if progbar:
            from ..utils import progbar as _progbar

            gates = _progbar(gates)

        for gate in gates:
            if isinstance(gate, Gate):
                self._apply_gate(gate, **gate_opts)
            else:
                self.apply_gate(*gate, **gate_opts)

        self._psi.squeeze_()

    def h(self, i, gate_round=None, **kwargs):
        self.apply_gate("H", i, gate_round=gate_round, **kwargs)

    def x(self, i, gate_round=None, **kwargs):
        self.apply_gate("X", i, gate_round=gate_round, **kwargs)

    def y(self, i, gate_round=None, **kwargs):
        self.apply_gate("Y", i, gate_round=gate_round, **kwargs)

    def z(self, i, gate_round=None, **kwargs):
        self.apply_gate("Z", i, gate_round=gate_round, **kwargs)

    def s(self, i, gate_round=None, **kwargs):
        self.apply_gate("S", i, gate_round=gate_round, **kwargs)

    def sdg(self, i, gate_round=None, **kwargs):
        self.apply_gate("SDG", i, gate_round=gate_round, **kwargs)

    def t(self, i, gate_round=None, **kwargs):
        self.apply_gate("T", i, gate_round=gate_round, **kwargs)

    def tdg(self, i, gate_round=None, **kwargs):
        self.apply_gate("TDG", i, gate_round=gate_round, **kwargs)

    def sx(self, i, gate_round=None, **kwargs):
        self.apply_gate("SX", i, gate_round=gate_round, **kwargs)

    def sxdg(self, i, gate_round=None, **kwargs):
        self.apply_gate("SXDG", i, gate_round=gate_round, **kwargs)

    def x_1_2(self, i, gate_round=None, **kwargs):
        self.apply_gate("X_1_2", i, gate_round=gate_round, **kwargs)

    def y_1_2(self, i, gate_round=None, **kwargs):
        self.apply_gate("Y_1_2", i, gate_round=gate_round, **kwargs)

    def z_1_2(self, i, gate_round=None, **kwargs):
        self.apply_gate("Z_1_2", i, gate_round=gate_round, **kwargs)

    def w_1_2(self, i, gate_round=None, **kwargs):
        self.apply_gate("W_1_2", i, gate_round=gate_round, **kwargs)

    def hz_1_2(self, i, gate_round=None, **kwargs):
        self.apply_gate("HZ_1_2", i, gate_round=gate_round, **kwargs)

    # constant two qubit gates

    def cnot(self, i, j, gate_round=None, **kwargs):
        self.apply_gate("CNOT", i, j, gate_round=gate_round, **kwargs)

    def cx(self, i, j, gate_round=None, **kwargs):
        self.apply_gate("CX", i, j, gate_round=gate_round, **kwargs)

    def cy(self, i, j, gate_round=None, **kwargs):
        self.apply_gate("CY", i, j, gate_round=gate_round, **kwargs)

    def cz(self, i, j, gate_round=None, **kwargs):
        self.apply_gate("CZ", i, j, gate_round=gate_round, **kwargs)

    def iswap(self, i, j, gate_round=None, **kwargs):
        self.apply_gate("ISWAP", i, j, **kwargs)

    # special non-tensor gates

    def iden(self, i, gate_round=None):
        pass

    def swap(self, i, j, gate_round=None, **kwargs):
        self.apply_gate("SWAP", i, j, **kwargs)

    # parametrizable gates

    def rx(self, theta, i, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "RX",
            theta,
            i,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def ry(self, theta, i, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "RY",
            theta,
            i,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def rz(self, theta, i, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "RZ",
            theta,
            i,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def u3(
        self,
        theta,
        phi,
        lamda,
        i,
        gate_round=None,
        parametrize=False,
        **kwargs,
    ):
        self.apply_gate(
            "U3",
            theta,
            phi,
            lamda,
            i,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def u2(self, phi, lamda, i, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "U2",
            phi,
            lamda,
            i,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def u1(self, lamda, i, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "U1",
            lamda,
            i,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def phase(self, lamda, i, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "PHASE",
            lamda,
            i,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def cu3(
        self,
        theta,
        phi,
        lamda,
        i,
        j,
        gate_round=None,
        parametrize=False,
        **kwargs,
    ):
        self.apply_gate(
            "CU3",
            theta,
            phi,
            lamda,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def cu2(
        self, phi, lamda, i, j, gate_round=None, parametrize=False, **kwargs
    ):
        self.apply_gate(
            "CU2",
            phi,
            lamda,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def cu1(self, lamda, i, j, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "CU1",
            lamda,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def cphase(
        self, lamda, i, j, gate_round=None, parametrize=False, **kwargs
    ):
        self.apply_gate(
            "CPHASE",
            lamda,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def fsim(
        self, theta, phi, i, j, gate_round=None, parametrize=False, **kwargs
    ):
        self.apply_gate(
            "FSIM",
            theta,
            phi,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def fsimg(
        self,
        theta,
        zeta,
        chi,
        gamma,
        phi,
        i,
        j,
        gate_round=None,
        parametrize=False,
        **kwargs,
    ):
        self.apply_gate(
            "FSIMG",
            theta,
            zeta,
            chi,
            gamma,
            phi,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def givens(
        self, theta, i, j, gate_round=None, parametrize=False, **kwargs
    ):
        self.apply_gate(
            "GIVENS",
            theta,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def givens2(
        self, theta, phi, i, j, gate_round=None, parametrize=False, **kwargs
    ):
        self.apply_gate(
            "GIVENS2",
            theta,
            phi,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def xx_plus_yy(
        self, theta, beta, i, j, gate_round=None, parametrize=False, **kwargs
    ):
        self.apply_gate(
            "XXPLUSYY",
            theta,
            beta,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def xx_minus_yy(
        self, theta, beta, i, j, gate_round=None, parametrize=False, **kwargs
    ):
        self.apply_gate(
            "XXMINUSYY",
            theta,
            beta,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def rxx(self, theta, i, j, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "RXX",
            theta,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def ryy(self, theta, i, j, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "RYY",
            theta,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def rzz(self, theta, i, j, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "RZZ",
            theta,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def crx(self, theta, i, j, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "CRX",
            theta,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def cry(self, theta, i, j, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "CRY",
            theta,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def crz(self, theta, i, j, gate_round=None, parametrize=False, **kwargs):
        self.apply_gate(
            "CRZ",
            theta,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def su4(
        self,
        theta1,
        phi1,
        lamda1,
        theta2,
        phi2,
        lamda2,
        theta3,
        phi3,
        lamda3,
        theta4,
        phi4,
        lamda4,
        t1,
        t2,
        t3,
        i,
        j,
        gate_round=None,
        parametrize=False,
        **kwargs,
    ):
        self.apply_gate(
            "SU4",
            theta1,
            phi1,
            lamda1,
            theta2,
            phi2,
            lamda2,
            theta3,
            phi3,
            lamda3,
            theta4,
            phi4,
            lamda4,
            t1,
            t2,
            t3,
            i,
            j,
            gate_round=gate_round,
            parametrize=parametrize,
            **kwargs,
        )

    def ccx(self, i, j, k, gate_round=None, **kwargs):
        self.apply_gate("CCX", i, j, k, gate_round=gate_round, **kwargs)

    def ccnot(self, i, j, k, gate_round=None, **kwargs):
        self.apply_gate("CCNOT", i, j, k, gate_round=gate_round, **kwargs)

    def toffoli(self, i, j, k, gate_round=None, **kwargs):
        self.apply_gate("TOFFOLI", i, j, k, gate_round=gate_round, **kwargs)

    def ccy(self, i, j, k, gate_round=None, **kwargs):
        self.apply_gate("CCY", i, j, k, gate_round=gate_round, **kwargs)

    def ccz(self, i, j, k, gate_round=None, **kwargs):
        self.apply_gate("CCZ", i, j, k, gate_round=gate_round, **kwargs)

    def cswap(self, i, j, k, gate_round=None, **kwargs):
        self.apply_gate("CSWAP", i, j, k, gate_round=gate_round, **kwargs)

    def fredkin(self, i, j, k, gate_round=None, **kwargs):
        self.apply_gate("FREDKIN", i, j, k, gate_round=gate_round, **kwargs)

    @property
    def psi(self):
        """Tensor network representation of the wavefunction."""
        # make sure all same dtype and drop singlet dimensions
        psi = self._psi.copy()
        psi.squeeze_()
        if not self.convert_eager:
            # not converted yet
            self._maybe_convert(psi)
        return psi

    def get_uni(self, transposed=False):
        """Tensor network representation of the unitary operator (i.e. with
        the initial state removed).
        """
        U = self.psi

        if transposed:
            # rename the initial state rand_uuid bonds to 1D site inds
            ixmap = {
                self.ket_site_ind(i): self.bra_site_ind(i)
                for i in range(self.N)
            }
        else:
            ixmap = {}

        # the first `N` tensors should be the tensors of input state
        tids = tuple(U.tensor_map)[: self.N]
        for i, tid in enumerate(tids):
            t = U.pop_tensor(tid)
            (old_ix,) = t.inds

            if transposed:
                ixmap[old_ix] = f"k{i}"
            else:
                ixmap[old_ix] = f"b{i}"

        U.reindex_(ixmap)
        U.view_as_(
            TensorNetworkGenOperator,
            upper_ind_id=self._ket_site_ind_id,
            lower_ind_id=self._bra_site_ind_id,
        )

        return U

    @property
    def uni(self):
        import warnings

        warnings.warn(
            "In future the tensor network returned by ``circ.uni`` will not "
            "be transposed as it is currently, to match the expectation from "
            "``U = circ.uni.to_dense()`` behaving like ``U @ psi``. You can "
            "retain this behaviour with ``circ.get_uni(transposed=True)``.",
            FutureWarning,
        )
        return self.get_uni(transposed=True)

    def get_reverse_lightcone_tags(self, where):
        """Get the tags of gates in this circuit corresponding to the 'reverse'
        lightcone propagating backwards from registers in ``where``.

        Parameters
        ----------
        where : int or sequence of int
            The register or register to get the reverse lightcone of.

        Returns
        -------
        tuple[str]
            The sequence of gate tags (``GATE_{i}``, ...) corresponding to the
            lightcone.
        """
        if isinstance(where, numbers.Integral):
            cone = {where}
        else:
            cone = set(where)

        lightcone_tags = []

        for i, gate in reversed(tuple(enumerate(self._gates))):
            if gate.label == "IDEN":
                continue
            elif gate.controls:
                # TODO: only add if any *targets* in cone, requires changes
                # elsewhere to make sure tensors aren't then missing
                regs = {*gate.controls, *gate.qubits}
                if regs & cone:
                    lightcone_tags.append(self.gate_tag(i))
                    cone |= regs
            elif gate.label == "SWAP":
                i, j = gate.qubits
                i_in_cone = i in cone
                j_in_cone = j in cone
                if i_in_cone:
                    cone.add(j)
                else:
                    cone.discard(j)
                if j_in_cone:
                    cone.add(i)
                else:
                    cone.discard(i)
            else:
                regs = set(gate.qubits)
                if regs & cone:
                    lightcone_tags.append(self.gate_tag(i))
                    cone |= regs

        # initial state is always part of the lightcone
        lightcone_tags.append("PSI0")
        lightcone_tags.reverse()

        return tuple(lightcone_tags)

    def get_psi_reverse_lightcone(self, where, keep_psi0=False):
        """Get just the bit of the wavefunction in the reverse lightcone of
        sites in ``where`` - i.e. causally linked.

        Parameters
        ----------
        where : int, or sequence of int
            The sites to propagate the the lightcone back from, supplied to
            :meth:`~quimb.tensor.circuit.Circuit.get_reverse_lightcone_tags`.
        keep_psi0 : bool, optional
            Keep the tensors corresponding to the initial wavefunction
            regardless of whether they are outside of the lightcone.

        Returns
        -------
        psi_lc : TensorNetwork1DVector
        """
        if isinstance(where, numbers.Integral):
            where = (where,)

        psi = self.psi
        lightcone_tags = self.get_reverse_lightcone_tags(where)
        psi_lc = psi.select_any(lightcone_tags).view_like_(psi)

        if not keep_psi0:
            # these sites are in the lightcone regardless of being alone
            site_inds = set(map(psi.site_ind, where))

            for tid, t in tuple(psi_lc.tensor_map.items()):
                # get all tensors connected to this tensor (incld itself)
                neighbors = oset_union(psi_lc.ind_map[ix] for ix in t.inds)

                # lone tensor not attached to anything - drop it
                # but only if it isn't directly in the ``where`` region
                if (len(neighbors) == 1) and set(t.inds).isdisjoint(site_inds):
                    psi_lc.pop_tensor(tid)

        return psi_lc

    def clear_storage(self):
        """Clear all cached data."""
        self._storage.clear()
        self._sampled_conditionals.clear()
        self._marginal_storage_size = 0
        self._sample_n_gates = self.num_gates

    def _maybe_init_storage(self):
        # clear/create the cache if circuit has changed
        if self._sample_n_gates != self.num_gates:
            self.clear_storage()

    def get_psi_simplified(
        self, seq="ADCRS", atol=1e-12, equalize_norms=False
    ):
        """Get the full wavefunction post local tensor network simplification.

        Parameters
        ----------
        seq : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.

        Returns
        -------
        psi : TensorNetwork1DVector
        """
        self._maybe_init_storage()

        key = ("psi_simplified", seq, atol)
        if key in self._storage:
            return self._storage[key].copy()

        # we simplify and store a copy
        psi = self._psi.copy()
        psi.squeeze_()

        # make sure to keep all outer indices
        output_inds = tuple(map(psi.site_ind, range(self.N)))

        # simplify the state and cache it
        psi.full_simplify_(
            seq=seq,
            atol=atol,
            output_inds=output_inds,
            equalize_norms=equalize_norms,
        )
        self._storage[key] = psi

        # return a copy so we can modify it inplace
        return psi.copy()

    def get_rdm_lightcone_simplified(
        self,
        where,
        seq="ADCRS",
        atol=1e-12,
        equalize_norms=False,
    ):
        """Get a simplified TN of the norm of the wavefunction, with
        gates outside reverse lightcone of ``where`` cancelled, and physical
        indices within ``where`` preserved so that they can be fixed (sliced)
        or used as output indices.

        Parameters
        ----------
        where : int or sequence of int
            The region assumed to be the target density matrix essentially.
            Supplied to
            :meth:`~quimb.tensor.circuit.Circuit.get_reverse_lightcone_tags`.
        seq : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.

        Returns
        -------
        TensorNetwork
        """
        key = ("rdm_lightcone_simplified", tuple(sorted(where)), seq, atol)
        if key in self._storage:
            return self._storage[key].copy()

        ket_lc = self.get_psi_reverse_lightcone(where)

        k_inds = tuple(map(self.ket_site_ind, where))
        b_inds = tuple(map(self.bra_site_ind, where))

        bra_lc = ket_lc.conj().reindex(dict(zip(k_inds, b_inds)))
        rho_lc = bra_lc | ket_lc

        # don't want to simplify site indices in region away
        output_inds = b_inds + k_inds

        # # simplify the norm and cache it
        rho_lc.full_simplify_(
            seq=seq,
            atol=atol,
            output_inds=output_inds,
            equalize_norms=equalize_norms,
        )
        self._storage[key] = rho_lc

        # return a copy so we can modify it inplace
        return rho_lc.copy()

    def amplitude(
        self,
        b,
        optimize="auto-hq",
        simplify_sequence="ADCRS",
        simplify_atol=1e-12,
        simplify_equalize_norms=True,
        backend=None,
        dtype=None,
        rehearse=False,
    ):
        r"""Get the amplitude coefficient of bitstring ``b``.

        .. math::

            c_b = \langle b | \psi \rangle

        Parameters
        ----------
        b : str or sequence of int
            The bitstring to compute the transition amplitude for.
        optimize : str, optional
            Contraction path optimizer to use for the amplitude, can be a
            non-reusable path optimizer as only called once (though path won't
            be cached for later use in that case).
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        backend : str, optional
            Backend to perform the contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        rehearse : bool or "tn", optional
            If ``True``, generate and cache the simplified tensor network and
            contraction tree but don't actually perform the contraction.
            Returns a dict with keys ``"tn"`` and ``'tree'`` with the tensor
            network that will be contracted and the corresponding contraction
            tree if so.
        """
        self._maybe_init_storage()

        if len(b) != self.N:
            raise ValueError(
                f"Bit-string {b} length does not "
                f"match number of qubits {self.N}."
            )

        fs_opts = {
            "seq": simplify_sequence,
            "atol": simplify_atol,
            "equalize_norms": simplify_equalize_norms,
        }

        # get the full wavefunction simplified
        psi_b = self.get_psi_simplified(**fs_opts)

        # fix the output indices to the correct bitstring
        for i, x in zip(range(self.N), b):
            psi_b.isel_({psi_b.site_ind(i): x})

        # perform a final simplification and cast
        psi_b.full_simplify_(**fs_opts)
        self._maybe_convert(psi_b, dtype)

        if rehearse == "tn":
            return psi_b

        tree = psi_b.contraction_tree(output_inds=(), optimize=optimize)

        if rehearse:
            return rehearsal_dict(psi_b, tree)

        # perform the full contraction with the tree found
        c_b = psi_b.contract(
            all, output_inds=(), optimize=tree, backend=backend
        )

        return c_b

    def amplitude_rehearse(
        self,
        b="random",
        simplify_sequence="ADCRS",
        simplify_atol=1e-12,
        simplify_equalize_norms=True,
        optimize="auto-hq",
        dtype=None,
        rehearse=True,
    ):
        """Perform just the tensor network simplifications and contraction tree
        finding associated with computing a single amplitude (caching the
        results) but don't perform the actual contraction.

        Parameters
        ----------
        b : 'random', str or sequence of int
            The bitstring to rehearse computing the transition amplitude for,
            if ``'random'`` (the default) a random bitstring will be used.
        optimize : str, optional
            Contraction path optimizer to use for the marginal, can be a
            non-reusable path optimizer as only called once (though path won't
            be cached for later use in that case).
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.

        Returns
        -------
        dict

        """
        if b == "random":
            b = "r" * self.N

        return self.amplitude(
            b=b,
            optimize=optimize,
            dtype=dtype,
            rehearse=rehearse,
            simplify_sequence=simplify_sequence,
            simplify_atol=simplify_atol,
            simplify_equalize_norms=simplify_equalize_norms,
        )

    amplitude_tn = functools.partialmethod(amplitude_rehearse, rehearse="tn")

    def partial_trace(
        self,
        keep,
        optimize="auto-hq",
        simplify_sequence="ADCRS",
        simplify_atol=1e-12,
        simplify_equalize_norms=True,
        backend=None,
        dtype=None,
        rehearse=False,
    ):
        r"""Perform the partial trace on the circuit wavefunction, retaining
        only qubits in ``keep``, and making use of reverse lightcone
        cancellation:

        .. math::

            \rho_{\bar{q}} = Tr_{\bar{p}}
            |\psi_{\bar{q}} \rangle \langle \psi_{\bar{q}}|

        Where :math:`\bar{q}` is the set of qubits to keep,
        :math:`\psi_{\bar{q}}` is the circuit wavefunction only with gates in
        the causal cone of this set, and :math:`\bar{p}` is the remaining
        qubits.

        Parameters
        ----------
        keep : int or sequence of int
            The qubit(s) to keep as we trace out the rest.
        optimize : str, optional
            Contraction path optimizer to use for the reduced density matrix,
            can be a non-reusable path optimizer as only called once (though
            path won't be cached for later use in that case).
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        rehearse : bool or "tn", optional
            If ``True``, generate and cache the simplified tensor network and
            contraction tree but don't actually perform the contraction.
            Returns a dict with keys ``"tn"`` and ``'tree'`` with the tensor
            network that will be contracted and the corresponding contraction
            tree if so.

        Returns
        -------
        array or dict
        """

        if isinstance(keep, numbers.Integral):
            keep = (keep,)

        output_inds = tuple(map(self.ket_site_ind, keep)) + tuple(
            map(self.bra_site_ind, keep)
        )

        rho = self.get_rdm_lightcone_simplified(
            where=keep,
            seq=simplify_sequence,
            atol=simplify_atol,
            equalize_norms=simplify_equalize_norms,
        )
        self._maybe_convert(rho, dtype)

        if rehearse == "tn":
            return rho

        tree = rho.contraction_tree(output_inds=output_inds, optimize=optimize)

        if rehearse:
            return rehearsal_dict(rho, tree)

        # perform the full contraction with the tree found
        rho_dense = rho.contract(
            all,
            output_inds=output_inds,
            optimize=tree,
            backend=backend,
        ).data

        return ops.reshape(rho_dense, [2 ** len(keep), 2 ** len(keep)])

    partial_trace_rehearse = functools.partialmethod(
        partial_trace, rehearse=True
    )
    partial_trace_tn = functools.partialmethod(partial_trace, rehearse="tn")

    def local_expectation(
        self,
        G,
        where,
        optimize="auto-hq",
        simplify_sequence="ADCRS",
        simplify_atol=1e-12,
        simplify_equalize_norms=True,
        backend=None,
        dtype=None,
        rehearse=False,
    ):
        r"""Compute the a single expectation value of operator ``G``, acting on
        sites ``where``, making use of reverse lightcone cancellation.

        .. math::

            \langle \psi_{\bar{q}} | G_{\bar{q}} | \psi_{\bar{q}} \rangle

        where :math:`\bar{q}` is the set of qubits :math:`G` acts one and
        :math:`\psi_{\bar{q}}` is the circuit wavefunction only with gates in
        the causal cone of this set. If you supply a tuple or list of gates
        then the expectations will be computed simultaneously.

        Parameters
        ----------
        G : array or sequence[array]
            The raw operator(s) to find the expectation of.
        where : int or sequence of int
            Which qubits the operator acts on.
        optimize : str, optional
            Contraction path optimizer to use for the local expectation,
            can be a non-reusable path optimizer as only called once (though
            path won't be cached for later use in that case).
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        gate_opts : None or dict_like
            Options to use when applying ``G`` to the wavefunction.
        rehearse : bool or "tn", optional
            If ``True``, generate and cache the simplified tensor network and
            contraction tree but don't actually perform the contraction.
            Returns a dict with keys ``'tn'`` and ``'tree'`` with the tensor
            network that will be contracted and the corresponding contraction
            tree if so.

        Returns
        -------
        scalar, tuple[scalar] or dict
        """
        if isinstance(where, numbers.Integral):
            where = (where,)

        fs_opts = {
            "seq": simplify_sequence,
            "atol": simplify_atol,
            "equalize_norms": simplify_equalize_norms,
        }

        rho = self.get_rdm_lightcone_simplified(where=where, **fs_opts)
        k_inds = tuple(self.ket_site_ind(i) for i in where)
        b_inds = tuple(self.bra_site_ind(i) for i in where)

        if isinstance(G, (list, tuple)):
            # if we have multiple expectations create an extra indexed stack
            nG = len(G)
            G_data = do("stack", G)
            G_data = reshape(G_data, (nG,) + (2,) * 2 * len(where))
            output_inds = (rand_uuid(),)
        else:
            G_data = reshape(G, (2,) * 2 * len(where))
            output_inds = ()

        TG = Tensor(data=G_data, inds=output_inds + b_inds + k_inds)

        rhoG = rho | TG

        rhoG.full_simplify_(output_inds=output_inds, **fs_opts)
        self._maybe_convert(rhoG, dtype)

        if rehearse == "tn":
            return rhoG

        tree = rhoG.contraction_tree(
            output_inds=output_inds, optimize=optimize
        )

        if rehearse:
            return rehearsal_dict(rhoG, tree)

        g_ex = rhoG.contract(
            all,
            output_inds=output_inds,
            optimize=tree,
            backend=backend,
        )

        if isinstance(g_ex, Tensor):
            g_ex = tuple(g_ex.data)

        return g_ex

    local_expectation_rehearse = functools.partialmethod(
        local_expectation, rehearse=True
    )
    local_expectation_tn = functools.partialmethod(
        local_expectation, rehearse="tn"
    )

    def compute_marginal(
        self,
        where,
        fix=None,
        optimize="auto-hq",
        backend=None,
        dtype="complex64",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
        rehearse=False,
    ):
        """Compute the probability tensor of qubits in ``where``, given
        possibly fixed qubits in ``fix`` and tracing everything else having
        removed redundant unitary gates.

        Parameters
        ----------
        where : sequence of int
            The qubits to compute the marginal probability distribution of.
        fix : None or dict[int, str], optional
            Measurement results on other qubits to fix.
        optimize : str, optional
            Contraction path optimizer to use for the marginal, can be a
            non-reusable path optimizer as only called once (though path won't
            be cached for later use in that case).
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        rehearse : bool or "tn", optional
            Whether to perform the marginal contraction or just return the
            associated TN and contraction tree.
        """
        self._maybe_init_storage()

        # index trick to contract straight to reduced density matrix diagonal
        # rho_ii -> p_i (i.e. insert a COPY tensor into the norm)
        output_inds = [self.ket_site_ind(i) for i in where]

        fs_opts = {
            "seq": simplify_sequence,
            "atol": simplify_atol,
            "equalize_norms": simplify_equalize_norms,
        }

        # lightcone region is target qubit plus fixed qubits
        region = set(where)
        if fix is not None:
            region |= set(fix)
        region = tuple(sorted(region))

        # have we fixed or are measuring all qubits?
        final_marginal = len(region) == self.N

        # these both are cached and produce TN copies
        if final_marginal:
            # won't need to partially trace anything -> just need ket
            nm_lc = self.get_psi_simplified(**fs_opts)
        else:
            # can use lightcone cancellation on partially traced qubits
            nm_lc = self.get_rdm_lightcone_simplified(region, **fs_opts)
            # re-connect the ket and bra indices as taking diagonal
            nm_lc.reindex_(
                {self.bra_site_ind(i): self.ket_site_ind(i) for i in region}
            )

        if fix:
            # project (slice) fixed tensors with bitstring
            # this severs the indices connecting bra and ket on fixed sites
            nm_lc.isel_({self.ket_site_ind(i): b for i, b in fix.items()})

        # having sliced we can do a final simplify
        nm_lc.full_simplify_(output_inds=output_inds, **fs_opts)

        # for stability with very small probabilities, scale by average prob
        if fix is not None:
            nfact = 2 ** len(fix)
            if final_marginal:
                nm_lc.multiply_(nfact**0.5, spread_over="all")
            else:
                nm_lc.multiply_(nfact, spread_over="all")

        # cast to desired data type
        self._maybe_convert(nm_lc, dtype)

        if rehearse == "tn":
            return nm_lc

        # NB. the tree isn't *neccesarily* the same each time due to the post
        #     projection full simplify, however there is also the lower level
        #     contraction path cache if the structure generated *is* the same
        #     so still pretty efficient to just overwrite
        tree = nm_lc.contraction_tree(
            output_inds=output_inds,
            optimize=optimize,
        )

        if rehearse:
            return rehearsal_dict(nm_lc, tree)

        # perform the full contraction with the tree found
        p_marginal = abs(
            nm_lc.contract(
                all,
                output_inds=output_inds,
                optimize=tree,
                backend=backend,
            ).data
        )

        if final_marginal:
            # we only did half the ket contraction so need to square
            p_marginal = p_marginal**2

        if fix is not None:
            p_marginal = p_marginal / nfact

        return p_marginal

    compute_marginal_rehearse = functools.partialmethod(
        compute_marginal, rehearse=True
    )
    compute_marginal_tn = functools.partialmethod(
        compute_marginal, rehearse="tn"
    )

    def calc_qubit_ordering(self, qubits=None, method="greedy-lightcone"):
        """Get a order to measure ``qubits`` in, by greedily choosing whichever
        has the smallest reverse lightcone followed by whichever expands this
        lightcone *least*.

        Parameters
        ----------
        qubits : None or sequence of int
            The qubits to generate a lightcone ordering for, if ``None``,
            assume all qubits.

        Returns
        -------
        tuple[int]
            The order to 'measure' qubits in.
        """
        self._maybe_init_storage()

        if qubits is None:
            qubits = tuple(range(self.N))
        else:
            qubits = tuple(sorted(qubits))

        key = ("lightcone_ordering", method, qubits)

        # check the cache first
        if key in self._storage:
            return self._storage[key]

        if method == "greedy-lightcone":
            cone = set()
            lctgs = {
                i: set(self.get_reverse_lightcone_tags(i)) for i in qubits
            }

            order = []
            while lctgs:
                # get the next qubit which adds least num gates to lightcone
                next_qubit = min(lctgs, key=lambda i: len(lctgs[i] - cone))
                cone |= lctgs.pop(next_qubit)
                order.append(next_qubit)

        else:
            # use graph distance based hierachical clustering
            psi = self.get_psi_simplified("R")
            qubit_inds = tuple(map(psi.site_ind, qubits))
            tids = psi._get_tids_from_inds(qubit_inds, "any")
            matcher = re.compile(psi.site_ind_id.format(r"(\d+)"))
            order = []
            for tid in psi.compute_hierarchical_ordering(tids, method=method):
                t = psi.tensor_map[tid]
                for ind in t.inds:
                    for sq in matcher.findall(ind):
                        order.append(int(sq))

        order = self._storage[key] = tuple(order)
        return order

    def _parse_qubits_order(self, qubits=None, order=None):
        """Simply initializes the default of measuring all qubits, and the
        default order, or checks that ``order`` is a permutation of ``qubits``.
        """
        if qubits is None:
            qubits = range(self.N)
        if order is None:
            order = self.calc_qubit_ordering(qubits)
        elif set(qubits) != set(order):
            raise ValueError("``order`` must be a permutation of ``qubits``.")

        return qubits, order

    def _group_order(self, order, group_size=1):
        """Take the qubit ordering ``order`` and batch it in groups of size
        ``group_size``, sorting the qubits (for caching reasons) within each
        group.
        """
        return tuple(
            tuple(sorted(g)) for g in partition_all(group_size, order)
        )

    def get_qubit_distances(self, method="dijkstra", alpha=2):
        """Get a nested dictionary of qubit distances. This is computed from a
        graph representing qubit interactions. The graph has an edge between
        qubits if they are acted on by the same gate, and the distance-weight
        of the edge is exponentially small in the number of gates between them.

        Parameters
        ----------
        method : {'dijkstra', 'resistance'}, optional
            The method to use to compute the qubit distances. See
            :func:`networkx.all_pairs_dijkstra_path_length` and
            :func:`networkx.resistance_distance`.
        alpha : float, optional
            The distance weight between qubits is ``alpha**(num_gates - 1 )``.

        Returns
        -------
        dict[int, dict[int, float]]
            The distance between each pair of qubits, accessed like
            ``distances[q1][q2]``. If two qubits are not connected, the
            distance is missing.
        """
        import networkx as nx

        G = nx.Graph()
        for g in self.gates:
            for q1, q2 in itertools.combinations(g.qubits, 2):
                if G.has_edge(q1, q2):
                    G[q1][q2]["weight"] /= alpha
                else:
                    G.add_edge(q1, q2, weight=1)

        if method == "dijkstra":
            distances = dict(
                nx.all_pairs_dijkstra_path_length(G, weight="weight")
            )
        elif method == "resistance":
            distances = nx.resistance_distance(G, weight="weight")
        else:
            raise ValueError(f"Unknown method {method}.")

        return distances

    def reordered_gates_dfs_clustered(self):
        """Get the gates reordered by a depth first search traversal of the
        multi-qubit gate graph that greedily selects successive gates which
        are 'close' in graph distance, and shifts single qubit gates to be
        adjacent to multi-qubit gates where possible.
        """
        # first we make a directed graph of the multi-qubit gates
        successors = {}
        predecessors = {}
        single_qubit_stacks = {}
        single_qubit_predecessors = {}
        last_gates = {}
        queue = []

        for i, g in enumerate(self.gates):
            if g.total_qubit_count == 1:
                # lazily accumulate single qubit gates
                (q,) = g.qubits
                single_qubit_stacks.setdefault(q, []).append(i)

            else:
                pi = predecessors[i] = []
                sqpi = single_qubit_predecessors[i] = []

                for q in g.qubits:
                    # collect any single qubit gates acting on this qubit
                    sqpi.extend(single_qubit_stacks.pop(q, []))

                    if q in last_gates:
                        # qubit has already been acted on -> have an edge
                        h = last_gates[q]
                        # mark h as a predecessor of i
                        pi.append(h)
                        # mark i as a successor of h
                        successors.setdefault(h, []).append(i)

                    # mark qubit as acted on
                    last_gates[q] = i

                if len(pi) == 0:
                    # no predecessors -> is possible starting multiqubit gate
                    queue.append(i)

        # then we traverse the multi-qubit gates in a depth first, topological
        # order, breaking ties by minimizing the distance between active qubits
        distances = self.get_qubit_distances()

        def gate_distance(i, j):
            qis = self.gates[i].qubits
            qjs = self.gates[j].qubits
            return min(
                distances[q1].get(q2, float("inf")) for q1 in qis for q2 in qjs
            )

        # sort initial queue by qubit with smallest index
        queue.sort(key=lambda i: min(self.gates[i].qubits))
        new_gates = []

        while queue:
            i = queue.pop(0)

            # first flush any single qubit gates acting on the qubits of gate i
            new_gates.extend(
                self.gates[j] for j in single_qubit_predecessors.pop(i, [])
            )
            # then add the gate itself
            new_gates.append(self.gates[i])

            # then remove i as a predecessor of its successors
            for j in successors.pop(i, []):
                pj = predecessors[j]
                pj.remove(i)
                if not pj:
                    # j has no more predecessors -> can be added to queue
                    queue.append(j)

            # check if this is the last time q is acted on,
            # if so flush any remaining single qubit gates
            for q in self.gates[i].qubits:
                if last_gates[q] == i:
                    # qubit has been acted on for the last time
                    new_gates.extend(
                        self.gates[j] for j in single_qubit_stacks.pop(q, [])
                    )

            # sort the queue of possible next gates
            queue.sort(key=lambda k: gate_distance(i, k))

        # flush any remaining single qubit gates
        for q in sorted(single_qubit_stacks):
            new_gates.extend(self.gates[j] for j in single_qubit_stacks.pop(q))

        return new_gates

    def sample(
        self,
        C,
        qubits=None,
        order=None,
        group_size=10,
        max_marginal_storage=2**20,
        seed=None,
        optimize="auto-hq",
        backend=None,
        dtype="complex64",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
    ):
        r"""Sample the circuit given by ``gates``, ``C`` times, using lightcone
        cancelling and caching marginal distribution results. This is a
        generator. This proceeds as a chain of marginal computations.

        Assuming we have ``group_size=1``, and some ordering of the qubits,
        :math:`\{q_0, q_1, q_2, q_3, \ldots\}` we first compute:

        .. math::

            p(q_0) = \mathrm{diag} \mathrm{Tr}_{1, 2, 3,\ldots}
            | \psi_{0} \rangle \langle \psi_{0} |

        I.e. simply the probability distribution on a single qubit, conditioned
        on nothing. The subscript on :math:`\psi` refers to the fact that we
        only need gates from the causal cone of qubit 0.
        From this we can sample an outcome, either 0 or 1, if we
        call this :math:`r_0` we can then move on to the next marginal:

        .. math::

            p(q_1 | r_0) = \mathrm{diag} \mathrm{Tr}_{2, 3,\ldots}
            \langle r_0
            | \psi_{0, 1} \rangle \langle \psi_{0, 1} |
            r_0 \rangle

        I.e. the probability distribution of the next qubit, given our prior
        result. We can sample from this to get :math:`r_1`. Then we compute:

        .. math::

            p(q_2 | r_0 r_1) = \mathrm{diag} \mathrm{Tr}_{3,\ldots}
            \langle r_0 r_1
            | \psi_{0, 1, 2} \rangle \langle \psi_{0, 1, 2} |
            r_0 r_1 \rangle

        Eventually we will reach the 'final marginal', which we can compute as

        .. math::

            |\langle r_0 r_1 r_2 r_3 \ldots | \psi \rangle|^2

        since there is nothing left to trace out.

        Parameters
        ----------
        C : int
            The number of times to sample.
        qubits : None or sequence of int, optional
            Which qubits to measure, defaults (``None``) to all qubits.
        order : None or sequence of int, optional
            Which order to measure the qubits in, defaults (``None``) to an
            order based on greedily expanding the smallest reverse lightcone.
            If specified it should be a permutation of ``qubits``.
        group_size : int, optional
            How many qubits to group together into marginals, the larger this
            is the fewer marginals need to be computed, which can be faster at
            the cost of higher memory. The marginal themselves will each be
            of size ``2**group_size``.
        max_marginal_storage : int, optional
            The total cumulative number of marginal probabilites to cache, once
            this is exceeded caching will be turned off.
        seed : None or int, optional
            A random seed, passed to ``numpy.random.seed`` if given.
        optimize : str, optional
            Contraction path optimizer to use for the marginals, shouldn't be
            a non-reusable path optimizer as called on many different TNs.
            Passed to :func:`cotengra.array_contract_tree`.
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.

        Yields
        ------
        bitstrings : sequence of str
        """
        # init TN norms, contraction trees, and marginals
        self._maybe_init_storage()

        rng = np.random.default_rng(seed)

        # which qubits and an ordering e.g. (2, 3, 4, 5), (5, 3, 4, 2)
        qubits, order = self._parse_qubits_order(qubits, order)

        # group the ordering e.g. ((5, 3), (4, 2))
        groups = self._group_order(order, group_size)

        result = dict()
        for _ in range(C):
            for where in groups:
                # key - (tuple[int] where, tuple[tuple[int q, str b])
                # value  - marginal probability distribution of `where` given
                #     prior results, as an ndarray
                # e.g. ((2,), ((0, '0'), (1, '0'))): array([1., 0.]), means
                #     prob(qubit2='0')=1 given qubit0='0' and qubit1='0'
                #     prob(qubit2='1')=0 given qubit0='0' and qubit1='0'
                key = (where, tuple(sorted(result.items())))
                if key not in self._sampled_conditionals:
                    # compute p(qs=x | current bitstring)
                    p = self.compute_marginal(
                        where=where,
                        fix=result,
                        optimize=optimize,
                        backend=backend,
                        dtype=dtype,
                        simplify_sequence=simplify_sequence,
                        simplify_atol=simplify_atol,
                        simplify_equalize_norms=simplify_equalize_norms,
                    )
                    p = do("to_numpy", p).astype("float64")
                    p /= p.sum()

                    if self._marginal_storage_size <= max_marginal_storage:
                        self._sampled_conditionals[key] = p
                        self._marginal_storage_size += p.size
                else:
                    p = self._sampled_conditionals[key]

                # the sampled bitstring e.g. '1' or '001010101'
                b_where = sample_bitstring_from_prob_ndarray(p, seed=rng)

                # split back into individual qubit results
                for q, b in zip(where, b_where):
                    result[q] = b

            yield "".join(result[i] for i in qubits)
            result.clear()

    def sample_rehearse(
        self,
        qubits=None,
        order=None,
        group_size=10,
        result=None,
        optimize="auto-hq",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
        rehearse=True,
        progbar=False,
    ):
        """Perform the preparations and contraction tree findings for
        :meth:`~quimb.tensor.circuit.Circuit.sample`, caching various
        intermedidate objects, but don't perform the main contractions.

        Parameters
        ----------
        qubits : None or sequence of int, optional
            Which qubits to measure, defaults (``None``) to all qubits.
        order : None or sequence of int, optional
            Which order to measure the qubits in, defaults (``None``) to an
            order based on greedily expanding the smallest reverse lightcone.
        group_size : int, optional
            How many qubits to group together into marginals, the larger this
            is the fewer marginals need to be computed, which can be faster at
            the cost of higher memory. The marginal's size itself is
            exponential in ``group_size``.
        result : None or dict[int, str], optional
            Explicitly check the computational cost of this result, assumed to
            be all zeros if not given.
        optimize : str, optional
            Contraction path optimizer to use for the marginals, shouldn't be
            a non-reusable path optimizer as called on many different TNs.
            Passed to :func:`cotengra.array_contract_tree`.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        progbar : bool, optional
            Whether to show the progress of finding each contraction tree.

        Returns
        -------
        dict[tuple[int], dict]
            One contraction tree object per grouped marginal computation.
            The keys of the dict are the qubits the marginal is computed for,
            the values are a dict containing a representative simplified tensor
            network (key: 'tn') and the main contraction tree (key: 'tree').
        """
        # init TN norms, contraction trees, and marginals
        self._maybe_init_storage()
        qubits, order = self._parse_qubits_order(qubits, order)
        groups = self._group_order(order, group_size)

        if result is None:
            result = {q: "r" for q in qubits}

        fix = {}
        tns_and_trees = {}

        for where in _progbar(groups, disable=not progbar):
            tns_and_trees[where] = self.compute_marginal(
                where=where,
                fix=fix,
                optimize=optimize,
                simplify_sequence=simplify_sequence,
                simplify_atol=simplify_atol,
                simplify_equalize_norms=simplify_equalize_norms,
                rehearse=rehearse,
            )

            # set the result of qubit ``q`` arbitrarily
            for q in where:
                fix[q] = result[q]

        return tns_and_trees

    sample_tns = functools.partialmethod(sample_rehearse, rehearse="tn")

    def sample_chaotic(
        self,
        C,
        marginal_qubits,
        fix=None,
        max_marginal_storage=2**20,
        seed=None,
        optimize="auto-hq",
        backend=None,
        dtype="complex64",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
    ):
        r"""Sample from this circuit, *assuming* it to be chaotic. Which is to
        say, only compute and sample correctly from the final marginal,
        assuming that the distribution on the other qubits is uniform.
        Given ``marginal_qubits=5`` for instance, for each sample a random
        bit-string :math:`r_0 r_1 r_2 \ldots r_{N - 6}` for the remaining
        :math:`N - 5` qubits will be chosen, then the final marginal will be
        computed as

        .. math::

            p(q_{N-5}q_{N-4}q_{N-3}q_{N-2}q_{N-1}
            | r_0 r_1 r_2 \ldots r_{N-6})
            =
            |\langle r_0 r_1 r_2 \ldots r_{N - 6} | \psi \rangle|^2

        and then sampled from. Note the expression on the right hand side has
        5 open indices here and so is a tensor, however if ``marginal_qubits``
        is not too big then the cost of contracting this is very similar to
        a single amplitude.

        .. note::

            This method *assumes* the circuit is chaotic, if its not, then the
            samples produced will not be an accurate representation of the
            probability distribution.

        Parameters
        ----------
        C : int
            The number of times to sample.
        marginal_qubits : int or sequence of int
            The number of qubits to treat as marginal, or the actual qubits. If
            an int is given then the qubits treated as marginal will be
            ``circuit.calc_qubit_ordering()[:marginal_qubits]``.
        fix : None or dict[int, str], optional
            Measurement results on other qubits to fix. These will be randomly
            sampled if ``fix`` is not given or a qubit is missing.
        seed : None or int, optional
            A random seed, passed to ``numpy.random.seed`` if given.
        optimize : str, optional
            Contraction path optimizer to use for the marginal, can be a
            non-reusable path optimizer as only called once (though path won't
            be cached for later use in that case).
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.

        Yields
        ------
        str
        """
        # init TN norms, contraction trees, and marginals
        self._maybe_init_storage()
        qubits = tuple(range(self.N))

        rng = np.random.default_rng(seed)

        # choose which qubits to treat as marginal - ideally 'towards one side'
        #     to increase contraction efficiency
        if isinstance(marginal_qubits, numbers.Integral):
            marginal_qubits = self.calc_qubit_ordering()[:marginal_qubits]
        where = tuple(sorted(marginal_qubits))

        # we will uniformly sample, and post-select on, the remaining qubits
        fix_qubits = tuple(q for q in qubits if q not in where)

        result = dict()
        for _ in range(C):
            # generate a random bit-string for the fixed qubits
            for q in fix_qubits:
                if (fix is None) or (q not in fix):
                    result[q] = rng.choice(("0", "1"))
                else:
                    result[q] = fix[q]

            # compute the remaining marginal
            key = (where, tuple(sorted(result.items())))
            if key not in self._sampled_conditionals:
                p = self.compute_marginal(
                    where=where,
                    fix=result,
                    optimize=optimize,
                    backend=backend,
                    dtype=dtype,
                    simplify_sequence=simplify_sequence,
                    simplify_atol=simplify_atol,
                    simplify_equalize_norms=simplify_equalize_norms,
                )
                p = do("to_numpy", p).astype("float64")
                p /= p.sum()

                if self._marginal_storage_size <= max_marginal_storage:
                    self._sampled_conditionals[key] = p
                    self._marginal_storage_size += p.size
            else:
                p = self._sampled_conditionals[key]

            # sample a bit-string for the marginal qubits
            b_where = sample_bitstring_from_prob_ndarray(p)

            # split back into individual qubit results
            for q, b in zip(where, b_where):
                result[q] = b

            yield "".join(result[i] for i in qubits)
            result.clear()

    def sample_chaotic_rehearse(
        self,
        marginal_qubits,
        result=None,
        optimize="auto-hq",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
        dtype="complex64",
        rehearse=True,
    ):
        """Rehearse chaotic sampling (perform just the TN simplifications and
        contraction tree finding).

        Parameters
        ----------
        marginal_qubits : int or sequence of int
            The number of qubits to treat as marginal, or the actual qubits. If
            an int is given then the qubits treated as marginal will be
            ``circuit.calc_qubit_ordering()[:marginal_qubits]``.
        result : None or dict[int, str], optional
            Explicitly check the computational cost of this result, assumed to
            be all zeros if not given.
        optimize : str, optional
            Contraction path optimizer to use for the marginal, can be a
            non-reusable path optimizer as only called once (though path won't
            be cached for later use in that case).
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        dtype : str, optional
            Data type to cast the TN to before contraction.

        Returns
        -------
        dict[tuple[int], dict]
            The contraction path information for the main computation, the key
            is the qubits that formed the final marginal. The value is itself a
            dict with keys ``'tn'`` - a representative tensor network - and
            ``'tree'`` - the contraction tree.
        """

        # init TN norms, contraction trees, and marginals
        self._maybe_init_storage()
        qubits = tuple(range(self.N))

        if isinstance(marginal_qubits, numbers.Integral):
            marginal_qubits = self.calc_qubit_ordering()[:marginal_qubits]
        where = tuple(sorted(marginal_qubits))

        fix_qubits = tuple(q for q in qubits if q not in where)

        if result is None:
            fix = {q: "0" for q in fix_qubits}
        else:
            fix = {q: result[q] for q in fix_qubits}

        rehs = self.compute_marginal(
            where=where,
            fix=fix,
            optimize=optimize,
            simplify_sequence=simplify_sequence,
            simplify_atol=simplify_atol,
            simplify_equalize_norms=simplify_equalize_norms,
            dtype=dtype,
            rehearse=rehearse,
        )

        if rehearse == "tn":
            return rehs

        return {where: rehs}

    sample_chaotic_tn = functools.partialmethod(
        sample_chaotic_rehearse, rehearse="tn"
    )

    def get_gate_by_gate_circuits(self, group_size=10):
        """Get a sequence of circuits by partitioning the gates into groups
        such circuit `i + 1` acts on at most ``group_size`` new qubits compared
        to circuit `i`.

        Parameters
        ----------
        group_size : int, optional
            The maximum number of new qubits that can be acted on by a circuit
            compared to its predecessor.

        Returns
        -------
        Sequence[dict]
            A sequence of dicts, each with keys ``'circuit'`` and ``'where'``,
            where the former is a :class:`~quimb.tensor.circuit.Circuit` and
            the latter the tuple of new qubits that it acts on comparaed to
            the previous circuit.
        """
        circs = [self.__class__(self.N)]
        groups = []
        current_group = set()

        # this ensures that single qubit gates are always adjacent to
        # multi-qubit gates and will thus always be included in the same group
        gates = self.reordered_gates_dfs_clustered()

        for gate in gates:
            # if we were to add next gate, how many new qubits would we have?
            next_group = current_group.union(gate.qubits)
            if len(next_group) > group_size:
                # over the limit: flush a copy of the current circuit and group
                groups.append(tuple(sorted(current_group)))
                circs.append(circs[-1].copy())
                # start a new group
                current_group = set(gate.qubits)
            else:
                # add the gate to the current group
                current_group = next_group
            circs[-1].apply_gate(gate)

        # add the final group corresponding to circs[-1]
        groups.append(tuple(sorted(current_group)))

        return tuple({"circuit": c, "where": g} for c, g in zip(circs, groups))

    def sample_gate_by_gate(
        self,
        C,
        group_size=10,
        seed=None,
        max_marginal_storage=2**20,
        optimize="auto-hq",
        backend=None,
        dtype="complex64",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
    ):
        """Sample this circuit using the gate-by-gate method, where we 'evolve'
        a result bitstring by sequentially including more and more gates, at
        each step updating the result by computing a full conditional marginal.
        See "How to simulate quantum measurement without computing marginals"
        by Sergey Bravyi, David Gosset, Yinchen Liu
        (https://arxiv.org/abs/2112.08499). The overall complexity of this is
        guaranteed to be similar to that of computing a single amplitude which
        can be much better than the naive "qubit-by-qubit" (`.sample`) method.
        However, it requires evaluting a number of tensor networks that scales
        linearly with the number of gates which can offset any practical
        advantages for shallow circuits for example.

        Parameters
        ----------
        C : int
            The number of samples to generate.
        group_size : int, optional
            The maximum number of qubits that can be acted on by a circuit
            compared to its predecessor. This will be the dimension of the
            marginal computed at each step.
        seed : None or int, optional
            A random seed, passed to ``numpy.random.seed`` if given.
        max_marginal_storage : int, optional
            The total cumulative number of marginal probabilites to cache, once
            this is exceeded caching will be turned off.
        optimize : str, optional
            Contraction path optimizer to use for the marginals, shouldn't be
            a non-reusable path optimizer as called on many different TNs.
            Passed to :func:`cotengra.array_contract_tree`.
        backend : str, optional
            Backend to perform the marginal contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        rehearse : bool, optional
            If ``True``, generate and cache the simplified tensor network and
            contraction tree but don't actually perform the contraction.
            Returns a dict with keys ``'tn'`` and ``'tree'`` with the tensor
            network that will be contracted and the corresponding contraction
            tree if so.

        Yields
        ------
        str
        """
        self._maybe_init_storage()

        rng = np.random.default_rng(seed)

        key = ("gate_by_gate_circuits", group_size)
        try:
            circs_wheres = self._storage[key]
        except KeyError:
            circs_wheres = self.get_gate_by_gate_circuits(group_size)
            self._storage[key] = circs_wheres

        for _ in range(C):
            # start with all qubits in the |0> state
            result = {q: "0" for q in range(self.N)}

            for circ_where in circs_wheres:
                # get the next circuit and the new group of qubits
                circ_g = circ_where["circuit"]
                where = circ_where["where"]

                # remove the new group of qubits from our current result
                for q in where:
                    result.pop(q)

                # check if we have already computed the conditional
                key = (where, tuple(sorted(result.items())))

                if key not in circ_g._sampled_conditionals:
                    p = circ_g.compute_marginal(
                        where,
                        fix=result,
                        optimize=optimize,
                        backend=backend,
                        dtype=dtype,
                        simplify_sequence=simplify_sequence,
                        simplify_atol=simplify_atol,
                        simplify_equalize_norms=simplify_equalize_norms,
                    )
                    p /= p.sum()

                    if circ_g._marginal_storage_size <= max_marginal_storage:
                        circ_g._sampled_conditionals[key] = p
                        circ_g._marginal_storage_size += p.size
                else:
                    p = circ_g._sampled_conditionals[key]

                # sample a configuration for our new group
                b_where = sample_bitstring_from_prob_ndarray(p, seed=rng)

                # update the fixed qubits given new group result
                for q, qx in zip(where, b_where):
                    result[q] = qx

            yield "".join(result[i] for i in range(self.N))

    def sample_gate_by_gate_rehearse(
        self,
        group_size=10,
        optimize="auto-hq",
        dtype="complex64",
        simplify_sequence="ADCRS",
        simplify_atol=1e-6,
        simplify_equalize_norms=True,
        rehearse=True,
        progbar=False,
    ):
        """Perform the preparations and contraction tree findings for
        :meth:`~quimb.tensor.circuit.Circuit.sample_gate_by_gate`, caching
        various intermedidate objects, but don't perform the main contractions.

        Parameters
        ----------
        group_size : int, optional
            The maximum number of qubits that can be acted on by a circuit
            compared to its predecessor. This will be the dimension of the
            marginal computed at each step.
        optimize : str, optional
            Contraction path optimizer to use for the marginals, shouldn't be
            a non-reusable path optimizer as called on many different TNs.
            Passed to :func:`cotengra.array_contract_tree`.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        rehearse : True or "tn", optional
            If ``True``, generate and cache the simplified tensor network and
            contraction tree but don't actually perform the contraction. If
            "tn", only generate the simplified tensor networks.

        Returns
        -------
        Sequence[dict] or Sequence[TensorNetwork]
        """
        self._maybe_init_storage()

        key = ("gate_by_gate_circuits", group_size)
        try:
            circs_wheres = self._storage[key]
        except KeyError:
            circs_wheres = self.get_gate_by_gate_circuits(group_size)
            self._storage[key] = circs_wheres

        rehs = []
        result = {q: "0" for q in range(self.N)}

        for circs_wheres in _progbar(circs_wheres, disable=not progbar):
            # get the next circuit and the new group of qubits
            circ_g = circs_wheres["circuit"]
            where = circs_wheres["where"]

            # remove the new group of qubits from our current result
            for q in where:
                result.pop(q)

            r = circ_g.compute_marginal(
                where,
                fix=result,
                optimize=optimize,
                dtype=dtype,
                simplify_sequence=simplify_sequence,
                simplify_atol=simplify_atol,
                simplify_equalize_norms=simplify_equalize_norms,
                rehearse=rehearse,
            )

            if rehearse != "tn":
                r["where"] = where
                r["circuit"] = circ_g

            rehs.append(r)

            # update the fixed qubits with randomly rotated results so we
            # don't get zero probability networks when simplifying
            for q in where:
                result[q] = "r"

        return rehs

    sample_gate_by_gate_tns = functools.partialmethod(
        sample_gate_by_gate_rehearse, rehearse="tn"
    )

    def to_dense(
        self,
        reverse=False,
        optimize="auto-hq",
        simplify_sequence="R",
        simplify_atol=1e-12,
        simplify_equalize_norms=True,
        backend=None,
        dtype=None,
        rehearse=False,
    ):
        """Generate the dense representation of the final wavefunction.

        Parameters
        ----------
        reverse : bool, optional
            Whether to reverse the order of the subsystems, to match the
            convention of qiskit for example.
        optimize : str, optional
            Contraction path optimizer to use for the contraction, can be a
            non-reusable path optimizer as only called once (though path won't
            be cached for later use in that case).
        dtype : dtype or str, optional
            If given, convert the tensors to this dtype prior to contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_equalize_norms : bool, optional
            Actively renormalize tensor norms during simplification.
        backend : str, optional
            Backend to perform the contraction with, e.g. ``'numpy'``,
            ``'cupy'`` or ``'jax'``. Passed to ``cotengra``.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        rehearse : bool, optional
            If ``True``, generate and cache the simplified tensor network and
            contraction tree but don't actually perform the contraction.
            Returns a dict with keys ``'tn'`` and ``'tree'`` with the tensor
            network that will be contracted and the corresponding contraction
            tree if so.

        Returns
        -------
        psi : qarray
            The densely represented wavefunction with ``dtype`` data.
        """
        psi = self.get_psi_simplified(
            seq=simplify_sequence,
            atol=simplify_atol,
            equalize_norms=simplify_equalize_norms,
        )
        self._maybe_convert(psi, dtype)

        if rehearse == "tn":
            return psi

        output_inds = tuple(map(psi.site_ind, range(self.N)))
        if reverse:
            output_inds = output_inds[::-1]

        tree = psi.contraction_tree(output_inds=output_inds, optimize=optimize)

        if rehearse:
            return rehearsal_dict(psi, tree)

        # perform the full contraction with the path found
        psi_tensor = psi.contract(
            all,
            output_inds=output_inds,
            optimize=tree,
            backend=backend,
        ).data

        k = ops.reshape(psi_tensor, (-1, 1))

        if isinstance(k, np.ndarray):
            k = qu.qarray(k)

        return k

    to_dense_rehearse = functools.partialmethod(to_dense, rehearse=True)
    to_dense_tn = functools.partialmethod(to_dense, rehearse="tn")

    def simulate_counts(self, C, seed=None, reverse=False, **to_dense_opts):
        """Simulate measuring all qubits in the computational basis many times.
        Unlike :meth:`~quimb.tensor.circuit.Circuit.sample`, this generates all
        the samples simultaneously using the full wavefunction constructed from
        :meth:`~quimb.tensor.circuit.Circuit.to_dense`, then calling
        :func:`~quimb.calc.simulate_counts`.

        .. warning::

            Because this constructs the full wavefunction it always requires
            exponential memory in the number of qubits, regardless of circuit
            depth and structure.

        Parameters
        ----------
        C : int
            The number of 'experimental runs', i.e. total counts.
        seed : int, optional
            A seed for reproducibility.
        reverse : bool, optional
            Whether to reverse the order of the subsystems, to match the
            convention of qiskit for example.
        to_dense_opts
            Suppled to :meth:`~quimb.tensor.circuit.Circuit.to_dense`.

        Returns
        -------
        results : dict[str, int]
            The number of recorded counts for each
        """
        p_dense = self.to_dense(reverse=reverse, **to_dense_opts)
        return qu.simulate_counts(p_dense, C=C, seed=seed)

    def schrodinger_contract(self, *args, **contract_opts):
        ntensor = self._psi.num_tensors
        path = [(0, 1)] + [(0, i) for i in reversed(range(1, ntensor - 1))]
        return self.psi.contract(*args, optimize=path, **contract_opts)

    def xeb(
        self,
        samples_or_counts,
        cache=None,
        cache_maxsize=2**20,
        progbar=False,
        **amplitude_opts,
    ):
        """Compute the linear cross entropy benchmark (XEB) for samples or
        counts, amplitude per amplitude.

        Parameters
        ----------
        samples_or_counts : Iterable[str] or Dict[str, int]
            Either the raw bitstring samples or a dict mapping bitstrings to
            the number of counts observed.
        cache : dict, optional
            A dictionary to store the probabilities in, if not supplied
            ``quimb.utils.LRU(cache_maxsize)`` will be used.
        cache_maxsize, optional
            The maximum size of the cache to be used.
        progbar, optional
            Whether to show progress as the bitstrings are iterated over.
        amplitude_opts
            Supplied to :meth:`~quimb.tensor.circuit.Circuit.amplitude`.
        """
        try:
            it = samples_or_counts.items()
        except AttributeError:
            it = zip(samples_or_counts, itertools.repeat(1))

        if progbar:
            it = _progbar(it)

        M = 0
        psum = 0.0

        if cache is None:
            cache = LRU(cache_maxsize)

        for b, cnt in it:
            try:
                p = cache[b]
            except KeyError:
                p = cache[b] = abs(self.amplitude(b, **amplitude_opts)) ** 2
            psum += cnt * p
            M += cnt

        return (2**self.N) / M * psum - 1

    def xeb_ex(
        self,
        optimize="auto-hq",
        simplify_sequence="R",
        simplify_atol=1e-12,
        simplify_equalize_norms=True,
        dtype=None,
        backend=None,
        autojit=False,
        progbar=False,
        **contract_opts,
    ):
        """Compute the exactly expected XEB for this circuit. The main feature
        here is that if you supply a cotengra optimizer that searches for
        sliced indices then the XEB will be computed without constructing the
        full wavefunction.

        Parameters
        ----------
        optimize : str or PathOptimizer, optional
            Contraction path optimizer.
        simplify_sequence : str, optional
            Simplifications to apply to tensor network prior to contraction.
        simplify_sequence : str, optional
            Which local tensor network simplifications to perform and in which
            order, see
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        simplify_atol : float, optional
            The tolerance with which to compare to zero when applying
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.full_simplify`.
        dtype : str, optional
            Data type to cast the TN to before contraction.
        backend : str, optional
            Convert tensors to, and then use contractions from, this library.
        autojit : bool, optional
            Apply ``autoray.autojit`` to the contraciton and map-reduce.
        progbar : bool, optional
            Show progress in terms of number of wavefunction chunks processed.
        """
        # get potentially simplified TN of full wavefunction
        psi = self.to_dense_tn(
            simplify_sequence=simplify_sequence,
            simplify_atol=simplify_atol,
            simplify_equalize_norms=simplify_equalize_norms,
            dtype=dtype,
        )

        # find a possibly sliced contraction tree
        output_inds = tuple(map(psi.site_ind, range(self.N)))
        tree = psi.contraction_tree(optimize=optimize, output_inds=output_inds)

        arrays = psi.arrays
        if backend is not None:
            arrays = [do("array", x, like=backend) for x in arrays]

        # perform map-reduce style computation over output wavefunction chunks
        # so we don't need entire wavefunction in memory at same time
        chunks = tree.gen_output_chunks(
            arrays, autojit=autojit, **contract_opts
        )
        if progbar:
            chunks = _progbar(chunks, total=tree.nchunks)

        def f(chunk):
            return do("sum", do("abs", chunk) ** 4)

        if autojit:
            # since we convert the arrays above, the jit backend is
            # automatically inferred
            from autoray import autojit

            f = autojit(f)

        p2sum = functools.reduce(operator.add, map(f, chunks))
        return 2**self.N * p2sum - 1

    def update_params_from(self, tn):
        """Assuming ``tn`` is a tensor network with tensors tagged ``GATE_{i}``
        corresponding to this circuit (e.g. from ``circ.psi`` or ``circ.uni``)
        but with updated parameters, update the current circuit parameters and
        tensors with those values.

        This is an inplace modification of the ``Circuit``.

        Parameters
        ----------
        tn : TensorNetwork
            The tensor network to find the updated parameters from.
        """
        for i, gate in enumerate(self._gates):
            tag = self.gate_tag(i)
            t = tn[tag]

            # sanity check that tensor(s) `t` correspond to the correct gate
            if gate.tag not in get_tags(t):
                raise ValueError(
                    f"The tensor(s) correponding to gate {i} "
                    f"should be tagged with '{gate.tag}', got {t}."
                )

            # only update gates and tensors if they are parametrizable
            if isinstance(t, PTensor):
                # update the actual tensor
                self._psi[tag].params = t.params

                # update the circuit's gate record
                self._gates[i] = Gate(
                    label=gate.label,
                    params=t.params,
                    qubits=gate.qubits,
                    round=gate.round,
                    parametrize=True,
                )

        self.clear_storage()

    def draw(
        self,
        figsize=None,
        radius=1 / 3,
        drawcolor=(0.5, 0.5, 0.5),
        linewidth=1,
    ):
        """Draw a simple linear schematic of the circuit.

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure, if not given will be set based on the
            number of gates and qubits.
        radius : float, optional
            The radius of the gates.
        drawcolor : tuple, optional
            The color of the wires.
        linewidth : float, optional
            The linewidth of the wires.

        Returns
        -------
        fig : matplotlib.Figure
            The figure object.
        ax : matplotlib.Axes
            The axis object.
        """
        from quimb.schematic import Drawing, hash_to_color

        if figsize is None:
            figsize = (self.num_gates / 6, self.N / 6)

        d = Drawing(
            figsize=figsize,
            presets=dict(
                wire=dict(
                    color=drawcolor,
                    linewidth=linewidth,
                ),
                gate=dict(
                    radius=radius,
                ),
            ),
        )

        depths = {}
        for i, g in enumerate(self.gates):
            # level = max(depths.get(q, 0) for q in g.qubits) + 1
            level = i

            if len(g.qubits) == 1:
                (q,) = g.qubits
                # draw line from previous gate to this one
                d.line(
                    (depths.get(q, -1) + radius, q),
                    (level - radius, q),
                    preset="wire",
                    zorder=level,
                )
                # draw the gate
                d.marker(
                    (level, q),
                    color=hash_to_color(g.label),
                    zorder=0,
                    preset="gate",
                )
                # record last gate on this qubit
                depths[q] = level
            else:
                # stretch a box over all qubits
                qmin = min(g.qubits)
                qmax = max(g.qubits)
                d.rectangle(
                    (level, qmin),
                    (level, qmax),
                    color=hash_to_color(g.label),
                    zorder=0,
                    alpha=1 / 3,
                    preset="gate",
                )
                for q in g.qubits:
                    # draw markers on each qubit acted on
                    d.marker(
                        (level, q),
                        color=hash_to_color(g.label),
                        zorder=0,
                        preset="gate",
                    )
                    # draw lines from previous gate to this one
                    d.line(
                        (depths.get(q, -1) + radius, q),
                        (level - radius, q),
                        preset="wire",
                        zorder=level,
                    )
                    # record last gate on this qubit
                    depths[q] = level

        # draw final lines to the right
        level = max(depths.values(), default=0) + 1
        for q in depths:
            d.line((depths.get(q, -1), q), (level, q), preset="wire")

        return d.fig, d.ax

    def __repr__(self):
        r = "<Circuit(n={}, num_gates={}, gate_opts={})>"
        return r.format(self.N, self.num_gates, self.gate_opts)


class CircuitMPS(Circuit):
    """Quantum circuit simulation keeping the state always in a MPS form. If
    you think the circuit will not build up much entanglement, or you just want
    to keep a rigorous handle on how much entanglement is present, this can
    be useful.

    Parameters
    ----------
    N : int, optional
        The number of qubits in the circuit.
    psi0 : TensorNetwork1DVector, optional
        The initial state, assumed to be ``|00000....0>`` if not given. The
        state is always copied and the tag ``PSI0`` added.
    max_bond : int, optional
        The maximum bond dimension to truncate to when applying gates, if any.
        This is simply a shortcut for setting ``gate_opts['max_bond']``.
    cutoff : float, optional
        The singular value cutoff to use when truncating the state.
        This is simply a shortcut for setting ``gate_opts['cutoff']``.
    gate_opts : dict, optional
        Default options to pass to each gate, for example, "max_bond" and
        "cutoff" etc.
    gate_contract : str, optional
        The default method for applying gates. Relevant MPS options are:

        - ``'auto-mps'``: automatically choose a method that maintains the
          MPS form (default). This uses ``'swap+split'`` for 2-qubit gates
          and ``'nonlocal'`` for 3+ qubit gates.
        - ``'swap+split'``: swap nonlocal qubits to be next to each other,
          before applying the gate, then swapping them back
        - ``'nonlocal'``: turn the gate into a potentially nonlocal (sub) MPO
          and apply it directly. See :func:`tensor_network_1d_compress`.

    dtype : str, optional
        The data type to use for the state tensor.
    to_backend : callable, optional
        A function to convert tensor data to a particular backend.
    convert_eager : bool, optional
        Whether to eagerly perform dtype casting and application of
        `to_backend` as gates are supplied, or wait until after the necessary
        TNs for a particular task such as sampling are formed and simplified.
        Eager conversion (`convert_eager=True`) is the default mode for
        MPS simulation, unlike full contraction.
    circuit_opts
        Supplied to :class:`~quimb.tensor.circuit.Circuit`.

    Attributes
    ----------
    psi : MatrixProductState
        The current state of the circuit, always in MPS form.

    Examples
    --------

    Create a circuit object that always uses the "nonlocal" method for
    contracting in gates, and the "dm" compression method within that, using
    a large cutoff and maximum bond dimension::

        circ = qtn.CircuitMPS(
            N=56,
            gate_opts=dict(
                contract="nonlocal",
                method="dm",
                max_bond=1024,
                cutoff=1e-3,
            )
        )

    """

    def __init__(
        self,
        N=None,
        *,
        psi0=None,
        max_bond=None,
        cutoff=1e-10,
        gate_opts=None,
        gate_contract="auto-mps",
        dtype=None,
        to_backend=None,
        convert_eager=True,
        **circuit_opts,
    ):
        gate_opts = ensure_dict(gate_opts)
        gate_opts.setdefault("contract", gate_contract)
        gate_opts.setdefault("propagate_tags", False)
        gate_opts.setdefault("max_bond", max_bond)
        gate_opts.setdefault("cutoff", cutoff)
        # this is used to pass around the canonical form
        gate_opts.setdefault("info", {})

        circuit_opts.setdefault("tag_gate_numbers", False)
        circuit_opts.setdefault("tag_gate_rounds", False)
        circuit_opts.setdefault("tag_gate_labels", False)

        circuit_opts.setdefault("dtype", dtype)
        circuit_opts.setdefault("to_backend", to_backend)
        circuit_opts.setdefault("convert_eager", convert_eager)

        super().__init__(N, psi0, gate_opts, **circuit_opts)

    def _init_state(self, N, dtype="complex128"):
        return MPS_computational_state("0" * N, dtype=dtype)

    def apply_gates(self, gates, progbar=False, **gate_opts):
        if progbar:
            from ..utils import progbar as _progbar

            gates = tuple(gates)
            gates = _progbar(gates, total=len(gates))
            gates.set_description(
                f"max_bond={self._psi.max_bond()}, "
                f"error~={self.error_estimate():.3g}"
            )

        for gate in gates:
            gate = parse_to_gate(gate)
            self._apply_gate(gate, **gate_opts)

            if progbar and (gate.total_qubit_count >= 2):
                # these don't change for single qubit gates
                gates.set_description(
                    f"max_bond={self._psi.max_bond()}, "
                    f"error~={self.error_estimate():.3g}"
                )

    @property
    def psi(self):
        # no squeeze so that bond dims of 1 preserved
        psi = self._psi.copy()
        if not self.convert_eager:
            self._maybe_convert(psi)
        return psi

    @property
    def uni(self):
        raise ValueError(
            "You can't extract the circuit unitary TN from a ``CircuitMPS``."
        )

    def calc_qubit_ordering(self, qubits=None):
        """MPS already has a natural ordering."""
        if qubits is None:
            return tuple(range(self.N))
        else:
            return tuple(sorted(qubits))

    def get_psi_reverse_lightcone(self, where, keep_psi0=False):
        """Override ``get_psi_reverse_lightcone`` as for an MPS the lightcone
        is not meaningful.
        """
        return self.psi

    def sample(
        self,
        C,
        seed=None,
        dtype=None,
        *,
        qubits=None,
        order=None,
        group_size=None,
        max_marginal_storage=None,
        optimize=None,
        backend=None,
        simplify_sequence=None,
        simplify_atol=None,
        simplify_equalize_norms=None,
    ):
        """Sample the MPS circuit ``C`` times.

        Parameters
        ----------
        C : int
            The number of samples to generate.
        seed : None, int, or generator, optional
            A random seed or generator to use for reproducibility.
        """
        unsupported = (
            qubits,
            order,
            group_size,
            max_marginal_storage,
            optimize,
            backend,
            simplify_sequence,
            simplify_atol,
            simplify_equalize_norms,
        )

        if any(x is not None for x in unsupported):
            warnings.warn(
                "Unsupported options for sampling an MPS circuit supplied, "
                "ignoring: " + ", ".join(map(str, unsupported))
            )

        if dtype is not None or not self.convert_eager:
            psi = self._psi.copy()
            self._maybe_convert(psi, dtype)
        else:
            psi = self._psi

        for config, _ in psi.sample(C, seed=seed):
            yield "".join(map(str, config))

    def fidelity_estimate(self):
        r"""Estimate the fidelity of the current state based on its norm, which
        tracks how much the state has been truncated:

        .. math::

            \tilde{F} =
            \left| \langle \psi | \psi \rangle \right|^2
            \approx
            \left|\langle \psi_\mathrm{ideal} | \psi \rangle\right|^2

        See Also
        --------
        error_estimate
        """
        cur_orthog = self.gate_opts["info"].get("cur_orthog", None)

        if cur_orthog is None:
            return abs(self._psi.norm()) ** 2

        cmin, cmax = cur_orthog
        return abs(self._psi[cmin : cmax + 1].norm(tags=all)) ** 2

    def error_estimate(self):
        r"""Estimate the error in the current state based on the norm of the
        discarded part of the state:

        .. math::

            \epsilon = 1 - \tilde{F}

        See Also
        --------
        fidelity_estimate
        """
        return 1 - self.fidelity_estimate()

    def local_expectation(
        self,
        G,
        where,
        normalized=False,
        dtype=None,
        *,
        simplify_sequence=None,
        simplify_atol=None,
        simplify_equalize_norms=None,
        backend=None,
        rehearse=None,
        **contract_opts,
    ):
        """Compute the local expectation value of a local operator at ``where``
        (via forming the reduced density matrix). Note this moves the
        orthogonality around inplace, and records it in `info`.

        Parameters
        ----------
        G : Tensor
            The local operator tensor.
        where : int
            The qubit to compute the expectation value at.
        normalized : bool, optional
            Whether to normalize the expectation value by the norm of the
            state.
        dtype : dtype, optional
            If given, ensure the TN is cast to this dtype before contracting.

        Returns
        -------
        float
        """
        unsupported = (
            simplify_sequence,
            simplify_atol,
            simplify_equalize_norms,
            backend,
            rehearse,
        )

        if any(x is not None for x in unsupported):
            warnings.warn(
                "Unsupported options for computing local_expectation with an "
                "MPS circuit supplied, ignoring: "
                + ", ".join(map(str, unsupported))
            )

        if dtype is not None or not self.convert_eager:
            psi = self._psi.copy()
            self._maybe_convert(psi, dtype)
        else:
            psi = self._psi

        return psi.local_expectation_canonical(
            G,
            where,
            normalized=normalized,
            info=self.gate_opts["info"],
            **contract_opts,
        )


class CircuitPermMPS(CircuitMPS):
    """Quantum circuit simulation keeping the state always in an MPS form, but
    lazily tracking the qubit ordering rather than 'swapping back' qubits after
    applying non-local gates. This can be useful for circuits with no
    expectation of locality. The qubit ordering is always tracked in the
    attribute ``qubits``. The ``psi`` attribute returns the TN with the sites
    reindexed and retagged according to the current qubit ordering, meaning it
    is no longer an MPS. Use `circ.get_psi_unordered()` to get the unpermuted
    MPS and use `circ.qubits` to get the current qubit ordering if you prefer.
    """

    def __init__(
        self,
        N=None,
        psi0=None,
        gate_opts=None,
        gate_contract="swap+split",
        **circuit_opts,
    ):
        gate_opts = ensure_dict(gate_opts)
        gate_opts.setdefault("contract", gate_contract)
        # this is used to pass around the canonical form
        gate_opts.setdefault("info", {})
        super().__init__(N, psi0=psi0, gate_opts=gate_opts, **circuit_opts)
        # keep track of the current qubit ordering
        self.qubits = list(range(self.N))

    def _apply_gate(self, gate, tags=None, **gate_opts):
        # first translate gate qubits to their current 'physical' location
        qubits = gate.qubits
        phys_sites = [self.qubits.index(q) for q in qubits]
        gate = gate.copy_with(qubits=phys_sites)

        # if the gate is non-local, account for swap (without swap back)
        if len(phys_sites) == 2:
            i, j = sorted(phys_sites)
            q = self.qubits.pop(j)
            self.qubits.insert(i + 1, q)
            gate_opts["swap_back"] = False

        super()._apply_gate(gate, tags=tags, **gate_opts)

    def calc_qubit_ordering(self, qubits=None):
        """Given by the current qubit permutation."""
        if qubits is None:
            return tuple(self.qubits)
        else:
            return tuple(sorted(qubits, key=self.qubits.index))

    def get_psi_unordered(self):
        """Return the MPS representing the state but without reordering the
        sites.
        """
        return self._psi.copy()

    def sample(
        self,
        C,
        seed=None,
        dtype=None,
    ):
        """Sample the PermMPS circuit ``C`` times.

        Parameters
        ----------
        C : int
            The number of samples to generate.
        seed : None, int, or generator, optional
            A random seed or generator to use for reproducibility.

        Yields
        ------
        str
            The next sample bitstring.
        """
        if dtype is not None or not self.convert_eager:
            psi = self._psi.copy()
            self._maybe_convert(psi, dtype)
        else:
            psi = self._psi

        # configurations are sampled in physical order, so invert the current
        # physical-site-to-logical-qubit mapping for logical bitstring output
        site_from_qubit = {
            qubit: site for site, qubit in enumerate(self.qubits)
        }
        for config, _ in psi.sample(C, seed=seed):
            yield "".join(
                str(config[site_from_qubit[i]]) for i in range(self.N)
            )

    @property
    def psi(self):
        # need to reindex and retag the MPS
        psi = self._psi.copy()

        psi.view_as_(TensorNetworkGenVector)
        psi.reindex_(
            {
                psi.site_ind(i): psi.site_ind(q)
                for i, q in enumerate(self.qubits)
            }
        )
        psi.retag_(
            {
                psi.site_tag(i): psi.site_tag(q)
                for i, q in enumerate(self.qubits)
            }
        )

        if not self.convert_eager:
            self._maybe_convert(psi)

        return psi


class CircuitDense(Circuit):
    """Quantum circuit simulation keeping the state in full dense form."""

    def __init__(
        self,
        N=None,
        psi0=None,
        gate_opts=None,
        gate_contract=True,
        tags=None,
        convert_eager=True,
        **circuit_opts,
    ):
        gate_opts = ensure_dict(gate_opts)
        gate_opts.setdefault("contract", gate_contract)
        gate_opts.setdefault("convert_eager", convert_eager)
        super().__init__(N, psi0, gate_opts, tags, **circuit_opts)

    @property
    def psi(self):
        t = self._psi ^ ...
        psi = t.as_network()
        psi.view_as_(Dense1D, like=self._psi)
        return psi

    @property
    def uni(self):
        raise ValueError(
            "You can't extract the circuit unitary TN from a ``CircuitDense``."
        )

    def calc_qubit_ordering(self, qubits=None):
        """Qubit ordering doesn't matter for a dense wavefunction."""
        if qubits is None:
            return tuple(range(self.N))
        else:
            return tuple(sorted(qubits))

    def get_psi_reverse_lightcone(self, where, keep_psi0=False):
        """Override ``get_psi_reverse_lightcone`` as for a dense wavefunction
        the lightcone is not meaningful.
        """
        return self.psi


class CircuitPEPOSimpleUpdate(Circuit):
    r"""Simulate a circuit in the Heisenberg picture by evolving an observable
    *backwards* in time as an arbitrary geometry PEPO (projected entangled pair
    operator), rather than evolving a state forwards.

    Gates are only *recorded* as they are applied - no contraction takes place.
    The work is deferred until :meth:`local_expectation` is called, at which
    point the observable is initialized as a bond dimension 1 PEPO on the
    circuit geometry, and each recorded gate ``g`` is absorbed in
    reverse order as the sandwich :math:`O \rightarrow g^\dagger O g` using
    :func:`~quimb.tensor.tnag.core.tensor_network_ag_gate_simple`, which gauges
    the local tensors, applies the gate and compresses the affected bond. The
    evolved operator is then contracted with the initial computational state to
    give :math:`\langle 0 | U^\dagger G U | 0 \rangle`.

    Because the observable spreads only through its reverse 'lightcone', gates
    acting entirely outside the operator's current support are skipped - they
    would contribute :math:`g^\dagger I g = I` - and only the support region is
    contracted at the end. Local observables of shallow circuits are therefore
    cheap even on very large geometries.

    Parameters
    ----------
    edges : sequence[tuple[hashable, hashable]], optional
        The nearest neighbor geometry: which pairs of sites are bonded, and so
        where two site gates may act. ``(u, v)`` and ``(v, u)`` denote the same
        edge. If neither ``edges`` nor ``gates`` is given, the geometry is built
        up dynamically from the two site gates as they are applied. Supplying
        ``edges`` fixes the geometry up front and restricts gates to it.
    gates : sequence, optional
        Optionally infer a fixed geometry up front from the two site gates in
        this sequence. The gates are only inspected here, not recorded.
    max_bond : int, optional
        The maximum bond dimension to compress the operator to. ``None`` means
        no limit.
    cutoff : float, optional
        The singular value cutoff to use when compressing the operator.
    gauge_smudge : float, optional
        Added to the gauges before they are multiplied in and inverted, to
        avoid numerical issues.
    gauge_power : float, optional
        The power to raise the singular values to before gauging.
    phys_dim : int, optional
        The physical dimension of each site, 2 by default.
    dtype : str, optional
        The data type of the operator tensors.
    circuit_opts
        Supplied to :class:`~quimb.tensor.circuit.Circuit`.

    Attributes
    ----------
    edges : tuple[tuple[hashable, hashable]]
        The unique geometry edges.
    sites : tuple[hashable]
        The sorted sites appearing in ``edges``.

    Examples
    --------

    Measure a local observable in the state prepared by a circuit, without
    ever forming that state::

        edges = [(0, 1), (1, 2), (2, 3)]
        circ = qtn.CircuitPEPOSimpleUpdate(edges, max_bond=16)
        circ.apply_gates(gates)  # recorded, nothing computed yet
        circ.local_expectation(qu.pauli("Z"), 1)  # evolved and contracted

    The geometry can also be left implicit and built up from the gates::

        circ = qtn.CircuitPEPOSimpleUpdate(max_bond=16)
        circ.apply_gates(gates)  # records the gates and defines the geometry
        circ.local_expectation(qu.pauli("Z"), 1)

    See Also
    --------
    CircuitMPS, tensor_network_ag_gate_simple
    """

    def __init__(
        self,
        edges=None,
        *,
        gates=None,
        max_bond=None,
        cutoff=1e-10,
        gauge_smudge=1e-6,
        gauge_power=1.0,
        phys_dim=2,
        dtype="complex128",
        **circuit_opts,
    ):
        if (edges is None) and (gates is not None):
            # infer a fixed geometry from the two site gates
            edges = (
                g.qubits
                for g in map(parse_to_gate, gates)
                if len(g.qubits) == 2
            )

        # if neither was given the geometry is grown lazily from the gates
        self._fixed_geometry = edges is not None
        edges = () if edges is None else tuple(gen_unique_edges(edges))

        self.edges = edges
        self._edge_set = {frozenset(e) for e in edges}
        self._site_set = {s for e in edges for s in e}
        self.sites = self._sorted_sites()
        self.phys_dim = phys_dim
        self._op_dtype = dtype
        self._su_opts = {
            "max_bond": max_bond,
            "cutoff": cutoff,
            "smudge": gauge_smudge,
            "power": gauge_power,
        }

        # gates are recorded not applied, so per-gate tagging is unnecessary
        circuit_opts.setdefault("tag_gate_numbers", False)
        circuit_opts.setdefault("tag_gate_rounds", False)
        circuit_opts.setdefault("tag_gate_labels", False)

        super().__init__(N=len(self.sites), **circuit_opts)

    def _sorted_sites(self):
        try:
            return tuple(sorted(self._site_set))
        except TypeError:
            # sites not mutually comparable, keep an arbitrary stable order
            return tuple(self._site_set)

    def _register_sites(self, sites):
        if not self._site_set.issuperset(sites):
            self._site_set.update(sites)
            self.sites = self._sorted_sites()
            self.N = len(self.sites)

    def _register_edge(self, edge):
        self._register_sites(edge)
        key = frozenset(edge)
        if key not in self._edge_set:
            self._edge_set.add(key)
            self.edges = (*self.edges, tuple(edge))

    def _init_state(self, N, dtype="complex128"):
        # the base class expects a state tensor network; it is never evolved,
        # but supplies the site structure used to close the operator at the end
        zero = np.zeros(self.phys_dim, dtype=self._op_dtype)
        zero[0] = 1.0
        return TN_from_sites_product_state({site: zero for site in self.sites})

    def copy(self):
        """Copy the circuit, including its geometry and recorded gates."""
        new = super().copy()
        new._fixed_geometry = self._fixed_geometry
        new.edges = self.edges
        new.sites = self.sites
        new._edge_set = set(self._edge_set)
        new._site_set = set(self._site_set)
        new.phys_dim = self.phys_dim
        new._op_dtype = self._op_dtype
        new._su_opts = dict(self._su_opts)
        return new

    def _apply_gate(self, gate, tags=None, **gate_opts):
        # Heisenberg picture: just record the gate, deferring all computation
        if gate.controls:
            raise NotImplementedError(
                "Controlled gates are not supported by "
                "`CircuitPEPOSimpleUpdate`."
            )
        if gate.special:
            raise NotImplementedError(
                f"The special gate {gate.label!r} has no array form; supply "
                "it as a raw array instead."
            )

        where = gate.qubits
        if len(where) == 1:
            if self._fixed_geometry:
                if where[0] not in self._site_set:
                    raise ValueError(
                        f"Gate site {where[0]} is not in the geometry."
                    )
            else:
                self._register_sites(where)
        elif len(where) == 2:
            if self._fixed_geometry:
                if frozenset(where) not in self._edge_set:
                    raise ValueError(
                        f"Gate on {where} is not a declared nearest neighbor "
                        "edge."
                    )
            else:
                self._register_edge(where)
        else:
            raise ValueError(
                "Only one and two site gates are supported, but got "
                f"{len(where)} sites: {where}."
            )

        self._gates.append(gate)

    def _identity_pepo(self, extra_sites=()):
        """Build the identity operator as a bond dimension 1 PEPO on the
        circuit geometry. Any sites not connected by an edge (including those in
        ``extra_sites``) are added as lone identity tensors.
        """
        d = self.phys_dim
        eye = np.eye(d, dtype=self._op_dtype)

        def fill_fn(shape):
            # shape is (*bond_dims, d, d), every bond dimension being 1
            return eye.reshape((1,) * (len(shape) - 2) + (d, d))

        if self.edges:
            op = TN_from_edges_and_fill_fn(
                fill_fn,
                self.edges,
                D=1,
                phys_dim=d,
                site_ind_id=("k{}", "b{}"),
            )
        else:
            op = TensorNetworkGenOperator.new(
                sites=(),
                site_tag_id="I{}",
                upper_ind_id="k{}",
                lower_ind_id="b{}",
            )

        edge_sites = {s for e in self.edges for s in e}
        for site in (self._site_set | set(extra_sites)) - edge_sites:
            op |= Tensor(
                eye,
                inds=(op.upper_ind(site), op.lower_ind(site)),
                tags=(op.site_tag(site),),
            )

        return op

    def _parse_where(self, where):
        try:
            single = where in self._site_set
        except TypeError:
            single = False
        if single:
            return (where,)

        if not isinstance(where, (tuple, list)):
            # a single hashable site not currently in the geometry
            if self._fixed_geometry:
                raise ValueError(
                    f"Observable site {where} is not in the geometry."
                )
            # with dynamic geometry an untouched site is allowed
            return (where,)

        where = tuple(where)
        if len(where) == 1:
            return self._parse_where(where[0])
        if len(where) > 2:
            raise ValueError(
                "Observables on more than two sites are not supported."
            )
        # a two site observable must lie on an existing edge
        if frozenset(where) not in self._edge_set:
            raise ValueError(
                f"Observable on {where} is not a nearest neighbor edge."
            )
        return where

    def _evolve_operator(self, G, where):
        """Build the observable ``G`` at ``where`` and evolve it backwards
        through all recorded gates, returning the compressed operator, its
        separately held bond gauges, an overall scale, and the support.
        """
        d = self.phys_dim
        op = self._identity_pepo(extra_sites=where)

        # seed the operator: G on the upper (ket) indices of the identity
        op.gate_upper_(
            np.asarray(G),
            where,
            contract=True if len(where) == 1 else "reduce-split",
        )

        gauges = {}
        scale = 1.0
        support = set(where)
        for gate in reversed(self._gates):
            qubits = gate.qubits
            if support.isdisjoint(qubits):
                # outside the reverse lightcone, g^dagger I g = I
                continue

            k = len(qubits)
            gdag = np.asarray(gate.array).reshape(d**k, d**k).conj().T

            # ``info`` exposes the single updated bond's unnormalized singular
            # values; with ``renorm=False`` these are kept, then the norm is
            # factored out into ``scale`` to keep the gauges and tensors O(1)
            info = {}
            op.gate_simple_(
                gdag,
                qubits,
                gauges,
                renorm=False,
                info=info,
                **self._su_opts,
            )
            if info:
                (_, ix), s = next(iter(info.items()))
                norm = float(do("linalg.norm", s))
                if norm > 0.0:
                    gauges[ix] = s / norm
                    scale = scale * norm

            support.update(qubits)

        return op, gauges, scale, support

    def local_expectation(self, G, where, *, optimize="auto", **contract_opts):
        r"""Compute :math:`\langle 0 | U^\dagger G U | 0 \rangle`, the
        expectation of the observable ``G`` at ``where`` in the state prepared
        by the recorded circuit ``U``, by evolving the operator backwards and
        contracting it with the initial all zeros state.

        Parameters
        ----------
        G : array_like
            The local observable, acting on the site(s) in ``where``.
        where : hashable or sequence[hashable]
            The site or sites the observable acts on. Each must be in the
            geometry, and a pair must be a declared edge.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer for the final contraction.
        contract_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract`.

        Returns
        -------
        scalar
        """
        where = self._parse_where(where)
        op, gauges, scale, support = self._evolve_operator(G, where)
        op.gauge_simple_insert(gauges)

        # only the reverse lightcone is non-identity; every other site gives
        # <0|I|0> = 1, so restrict the contraction to the support region
        tn = op.select(
            [op.site_tag(site) for site in support], which="any"
        ).copy()

        zero = np.zeros(self.phys_dim, dtype=self._op_dtype)
        zero[0] = 1.0
        for site in support:
            tn |= Tensor(zero, inds=(op.upper_ind(site),))
            tn |= Tensor(zero, inds=(op.lower_ind(site),))

        # close the remaining size 1 bonds to the dropped identity region
        tn.isel_({ix: 0 for ix in tn.outer_inds()})

        return scale * tn.contract(all, optimize=optimize, **contract_opts)

    def compute_local_expectation(
        self,
        terms,
        *,
        optimize="auto",
        return_all=False,
        progbar=False,
        **contract_opts,
    ):
        """Compute many local expectations, each evolved backwards
        independently.

        Parameters
        ----------
        terms : dict[hashable or sequence[hashable], array_like]
            Mapping of site(s) to the local observable acting there.
        optimize : str or PathOptimizer, optional
            The contraction path optimizer for the final contractions.
        return_all : bool, optional
            Whether to return all results keyed by location, or just their sum.
        progbar : bool, optional
            Whether to show a progress bar over the terms.
        contract_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.contract`.

        Returns
        -------
        scalar or dict[hashable or sequence[hashable], scalar]
        """
        terms = dict(terms)
        keys = tuple(terms)
        if progbar:
            keys = _progbar(keys)

        expecs = {
            where: self.local_expectation(
                terms[where], where, optimize=optimize, **contract_opts
            )
            for where in keys
        }
        if return_all:
            return expecs
        return sum(expecs.values())

    @property
    def psi(self):
        raise NotImplementedError(
            "`CircuitPEPOSimpleUpdate` works in the Heisenberg picture and "
            "keeps no explicit forward state; use `local_expectation`."
        )

    @property
    def uni(self):
        raise ValueError(
            "You can't extract the circuit unitary TN from a "
            "``CircuitPEPOSimpleUpdate``."
        )

    def _requires_forward_state(self, *_, **__):
        raise NotImplementedError(
            f"`{type(self).__name__}` does not form the explicit forward "
            "evolved state; use `local_expectation` instead."
        )

    to_dense = _requires_forward_state
    amplitude = _requires_forward_state
    partial_trace = _requires_forward_state
    sample = _requires_forward_state
    sample_chaotic = _requires_forward_state

    def calc_qubit_ordering(self, qubits=None):
        """The PEPO geometry has no inherent linear qubit ordering."""
        if qubits is None:
            return tuple(self.sites)
        return tuple(qubits)

    def __repr__(self):
        return (
            f"<CircuitPEPOSimpleUpdate("
            f"N={self.N}, "
            f"num_gates={self.num_gates}, "
            f"max_bond={self._su_opts['max_bond']})>"
        )


class CircuitPEPSSimpleUpdate(Circuit):
    """Quantum circuit simulation keeping the state as a generic tensor
    network (a "PEPS" defined by an arbitrary graph of ``edges``) and applying
    gates with the simple update rule. The state always keeps a single tensor
    per site, with bonds only along the supplied edges; two-qubit gates are
    only supported on those edges. Bond singular values are tracked as
    Vidal-style gauges, which makes gate application and the computation of
    local expectations cheap and approximate.

    This is useful for circuits on lattices that build up more than 1D worth of
    entanglement, where an exact or MPS simulation is intractable but a
    truncated, gauged tensor network state is a good approximation.

    Parameters
    ----------
    N : int, optional
        The number of qubits in the circuit. If not given it is inferred from
        the geometry. Supply it to pad the geometry up to ``N`` sites,
        including any that have no edges.
    edges : sequence[tuple[int, int]], optional
        The edges defining the geometry of the PEPS. A bond is placed between
        each pair of sites, and two-qubit gates are only supported on these
        edges. Every site appearing in ``edges`` is included. If not given the
        geometry is taken from ``gates`` or ``psi0`` instead.
    gates : sequence, optional
        If ``edges`` is not given, infer the geometry from the two-qubit gates
        in this sequence. The gates are only inspected here, not applied, so
        you still pass them to :meth:`apply_gates` afterwards.
    psi0 : TensorNetworkGenVector, optional
        Supply the initial state directly instead of starting from the
        ``|00...0>`` product state. If ``edges`` is not given the geometry is
        read from the bonds of this state, and the bond gauges are seeded from
        it. Only a single seeding sweep is performed; unlike imaginary time
        simple update the gauge matters immediately, so for an arbitrary
        ``psi0`` you may want to call :meth:`equilibrate` once before applying
        gates.
    max_bond : int, optional
        The maximum bond dimension to truncate to when applying gates.
    cutoff : float, optional
        The singular value cutoff to use when truncating after applying gates.
    renorm : bool, optional
        Whether to renormalize the singular values of a bond after each gate.
        The default ``False`` tracks the norm of the state rather than forcing
        it to one, which is the sensible choice for real time and general
        circuit dynamics. Set ``True`` to instead keep the state normalized
        after every gate, e.g. for the near-identity gates of imaginary time
        evolution.
    gauge_smudge : float, optional
        Small value added to the gauges before they are multiplied in and
        inverted, for numerical stability with very small singular values.
    equilibrate_every : int, optional
        If given, automatically call :meth:`equilibrate` after every this many
        gates have been applied.
    equilibrate_opts : dict, optional
        Default options forwarded to :meth:`equilibrate`.
    gate_opts : dict, optional
        Default options to pass to ``gate_simple_`` such as ``max_bond`` and
        ``cutoff``.
    dtype : str, optional
        If given, ensure the state tensors are cast to this data type.
    to_backend : callable, optional
        If given, apply this function to the state tensors to convert them to a
        particular array backend.
    convert_eager : bool, optional
        Whether to apply the ``dtype`` and ``to_backend`` conversions eagerly
        as each gate is applied. The default ``True`` matches the other running
        simulators (e.g. :class:`CircuitMPS`), since the simple update rule
        contracts each gate into the state immediately rather than building a
        lazy network to contract later.

    Attributes
    ----------
    edges : tuple[tuple[hashable, hashable]]
        The unique edges defining the PEPS geometry.
    sites : tuple[hashable]
        The sites (qubit labels) of the PEPS.
    gauges : dict[str, array]
        The current Vidal-style bond gauges (singular values), keyed by bond
        index, updated in place as gates are applied.

    Notes
    -----
    The gates applied must address qubits using the same labels that appear in
    ``edges``. Two-qubit gates are only supported along an existing edge.

    Examples
    --------

        >>> import quimb.tensor as qtn
        >>> edges = [(0, 1), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5)]
        >>> circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=8)
        >>> circ.apply_gates(gates)
        >>> peps = circ.psi

    See Also
    --------
    CircuitMPS, CircuitDense
    """

    def __init__(
        self,
        N=None,
        *,
        edges=None,
        gates=None,
        psi0=None,
        max_bond=None,
        cutoff=1e-10,
        renorm=False,
        gauge_smudge=1e-12,
        equilibrate_every=None,
        equilibrate_opts=None,
        gate_opts=None,
        dtype=None,
        to_backend=None,
        convert_eager=True,
        **circuit_opts,
    ):
        # geometry can come from explicit `edges`, be inferred from the two
        # site `gates` (only inspected here, not applied) or be read from the
        # bonds of an existing `psi0`
        extra_sites = ()
        if edges is None:
            if psi0 is not None:
                edges = tuple(psi0.gen_bond_coos())
            elif gates is not None:
                parsed = [parse_to_gate(g) for g in gates]
                edges = [g.qubits for g in parsed if len(g.qubits) == 2]
                extra_sites = tuple(q for g in parsed for q in g.qubits)
            else:
                raise ValueError(
                    "You must supply one of `edges`, `gates` or `psi0` to "
                    "define the PEPS geometry."
                )
        self._edges = tuple(gen_unique_edges(edges))

        # sites are everything appearing in the edges, plus any extra sites
        # touched by single qubit gates or present in psi0, padded up to N
        sites = set()
        for a, b in self._edges:
            sites.add(a)
            sites.add(b)
        sites.update(extra_sites)
        if psi0 is not None:
            sites.update(psi0.sites)
        if N is not None:
            sites.update(range(N))
        self._sites = tuple(sorted(sites))
        self._site_set = set(self._sites)
        self._edge_set = {frozenset(e) for e in self._edges}

        # bond gauges tracked across gate applications
        self.gauges = {}

        # auto re-gauge every this many gates, if given
        self._equilibrate_every = equilibrate_every
        self._equilibrate_opts = ensure_dict(equilibrate_opts)

        gate_opts = ensure_dict(gate_opts)
        gate_opts.setdefault("max_bond", max_bond)
        gate_opts.setdefault("cutoff", cutoff)
        gate_opts.setdefault("renorm", renorm)
        gate_opts.setdefault("smudge", gauge_smudge)

        circuit_opts.setdefault("tag_gate_numbers", False)
        circuit_opts.setdefault("tag_gate_rounds", False)
        circuit_opts.setdefault("tag_gate_labels", False)

        circuit_opts.setdefault("dtype", dtype)
        circuit_opts.setdefault("to_backend", to_backend)
        circuit_opts.setdefault("convert_eager", convert_eager)

        super().__init__(len(self._sites), psi0, gate_opts, **circuit_opts)

        if psi0 is not None:
            # seed the bond gauges from the supplied state
            self._psi.gauge_all_simple_(gauges=self.gauges, max_iterations=1)

    def copy(self):
        """Copy the circuit, including its state, gauges and geometry. The
        base :class:`Circuit` copy does not know about the extra simple update
        attributes, so they are carried over here (the gauges are copied so the
        two circuits can be evolved independently).
        """
        new = super().copy()
        new._edges = self._edges
        new._sites = self._sites
        new._site_set = self._site_set
        new._edge_set = self._edge_set
        new.gauges = dict(self.gauges)
        new._equilibrate_every = self._equilibrate_every
        new._equilibrate_opts = dict(self._equilibrate_opts)
        return new

    @property
    def edges(self):
        """The unique edges defining the PEPS geometry."""
        return self._edges

    @property
    def sites(self):
        """The sites (qubit labels) of the PEPS."""
        return self._sites

    def _init_state(self, N, dtype="complex128"):
        # |00...0> product state with bond dimension 1 bonds along the edges
        zero = do("array", [1.0, 0.0], dtype=dtype)
        psi = TN_from_sites_product_state({site: zero for site in self._sites})
        for a, b in self.edges:
            psi[a].new_bond(psi[b])
        return psi

    def _apply_gate(self, gate, tags=None, **gate_opts):
        # route gate application through the simple update rule, threading the
        # persistent bond gauges so they stay consistent between gates
        if gate.controls:
            raise ValueError(
                "Controlled gates are not supported by "
                "`CircuitPEPSSimpleUpdate`, since the simple update rule "
                "applies a dense gate array to the sites. Supply the gate as "
                "a full unitary on its qubits instead."
            )
        if gate.special:
            raise ValueError(
                f"The special gate {gate.label!r} is not supported by "
                "`CircuitPEPSSimpleUpdate`. Supply a gate with an explicit "
                "array acting on sites connected by an edge."
            )

        where = gate.qubits
        if len(where) > 2:
            raise ValueError(
                "`CircuitPEPSSimpleUpdate` only supports one and two site "
                f"gates, but got {len(where)} sites: {where}."
            )
        if (len(where) == 2) and (frozenset(where) not in self._edge_set):
            raise ValueError(
                f"The gate acts on sites {where} which are not a declared "
                "edge of the PEPS, only nearest neighbor gates are allowed."
            )

        opts = {**self.gate_opts, **gate_opts}
        opts.pop("contract", None)
        opts.pop("propagate_tags", None)

        G = gate.array
        if self.convert_eager:
            key = id(G)
            if key not in self._backend_gate_cache:
                self._backend_gate_cache[key] = self._maybe_convert(G)
            G = self._backend_gate_cache[key]

        self._psi.gate_simple_(G, where, self.gauges, **opts)
        self._gates.append(gate)

        if self._equilibrate_every and (
            len(self._gates) % self._equilibrate_every == 0
        ):
            self.equilibrate()

    def apply_gates(self, gates, progbar=False, **gate_opts):
        if progbar:
            from ..utils import progbar as _progbar

            gates = tuple(gates)
            gates = _progbar(gates, total=len(gates))
            gates.set_description(f"max_bond={self._psi.max_bond()}")

        for gate in gates:
            gate = parse_to_gate(gate)
            self._apply_gate(gate, **gate_opts)

            if progbar and (gate.total_qubit_count >= 2):
                gates.set_description(f"max_bond={self._psi.max_bond()}")

    def equilibrate(self, **gauge_opts):
        """Re-gauge the whole state with the simple update rule, improving the
        consistency of the tracked bond gauges. This does not change the state
        represented, only the gauge, and can be called periodically between
        rounds of gates to keep the simple update approximation well behaved.

        The default options given at construction via ``equilibrate_opts`` are
        applied first, with any keyword arguments here taking precedence.

        Parameters
        ----------
        gauge_opts
            Supplied to
            :meth:`~quimb.tensor.tensor_core.TensorNetwork.gauge_all_simple_`,
            for example ``max_iterations`` and ``tol``.
        """
        opts = {**self._equilibrate_opts, **gauge_opts}
        opts.setdefault("max_iterations", 100)
        opts.setdefault("tol", 1e-10)
        self._psi.gauge_all_simple_(gauges=self.gauges, **opts)

    def local_expectation(
        self,
        G,
        where,
        *,
        max_distance=0,
        normalized=True,
        **contract_opts,
    ):
        """Compute the local expectation value of operator ``G`` at the site(s)
        ``where``, using the simple update bond gauges to approximate the
        environment beyond ``max_distance``.

        Parameters
        ----------
        G : array_like
            The local operator.
        where : hashable or sequence[hashable]
            The site or sites to compute the expectation at. A single site
            label (which may itself be a tuple, e.g. a 2D coordinate) is
            detected by membership in the set of sites.
        max_distance : int, optional
            How many graph hops of neighboring tensors to include in the local
            cluster used to approximate the reduced density matrix. The default
            ``0`` uses only the target site(s) and their gauges, matching
            :meth:`~quimb.tensor.tnag.core.TensorNetworkGenVector.compute_local_expectation_cluster`.
        normalized : bool, optional
            Whether to normalize by the local norm.
        contract_opts
            Supplied to
            :meth:`~quimb.tensor.tnag.core.TensorNetworkGenVector.compute_local_expectation_cluster`.

        Returns
        -------
        float
        """
        if isinstance(where, list):
            where = tuple(where)
        if where in self._site_set:
            where = (where,)
        else:
            where = tuple(where)
        return self._psi.compute_local_expectation_cluster(
            {where: G},
            gauges=self.gauges,
            max_distance=max_distance,
            normalized=normalized,
            **contract_opts,
        )

    def get_state(self, absorb_gauges=True):
        """Return the current PEPS state, optionally absorbing the bond gauges.

        Parameters
        ----------
        absorb_gauges : bool or "return", optional
            How to handle the tracked Vidal-style bond gauges. If ``True`` (the
            default) the gauges are absorbed, so the returned tensor network is
            the actual wavefunction (up to the simple update approximation). If
            ``False`` the gauges are added to the network as uncontracted
            diagonal tensors. If ``"return"`` the raw gauged network and a copy
            of the gauges are returned separately. The internal state is left
            untouched in every case.

        Returns
        -------
        psi : TensorNetwork
            The current state.
        gauges : dict
            The current gauges, only if ``absorb_gauges == "return"``.
        """
        psi = self._psi.copy()

        if absorb_gauges == "return":
            gauges = dict(self.gauges)
            if not self.convert_eager:
                self._maybe_convert(psi)
            return psi, gauges

        if absorb_gauges:
            # absorb the Vidal-form bond gauges so the returned TN is the state
            psi.gauge_simple_insert(self.gauges)
        else:
            # add the gauges as uncontracted diagonal tensors on their bonds
            for ix, g in self.gauges.items():
                psi |= Tensor(g, inds=[ix])

        if not self.convert_eager:
            self._maybe_convert(psi)
        return psi

    @property
    def psi(self):
        """The PEPS tensor network state, with the simple update bond gauges
        absorbed back in so that it represents the actual wavefunction (a
        proper contraction of ``psi`` gives the state, up to the simple update
        approximation). The internal gauged form is left untouched. Shorthand
        for ``get_state(absorb_gauges=True)``.
        """
        return self.get_state(absorb_gauges=True)

    def calc_qubit_ordering(self, qubits=None):
        if qubits is None:
            return tuple(self._sites)
        return tuple(sorted(qubits))

    def _unsupported_exact(self, name):
        raise NotImplementedError(
            f"`{name}` is not supported by `CircuitPEPSSimpleUpdate`, which "
            "only ever holds an approximate, gauged tensor network state. Use "
            "`local_expectation` for observables or `psi` to get the gauged "
            "PEPS state and contract or sample it with the approximation you "
            "want."
        )

    def to_dense(self, *args, **kwargs):
        """Contract the gauged PEPS into a dense wavefunction, a column-vector
        ``qarray`` of length ``2**N`` ordered like :attr:`sites`, matching the
        output of :meth:`Circuit.to_dense`. This is the actual (approximate)
        state, so the cost grows exponentially with the number of qubits.

        Arguments are forwarded to
        :meth:`~quimb.tensor.tnag.core.TensorNetworkGenVector.to_dense`.
        """
        return self.psi.to_dense(*args, **kwargs)

    def amplitude(self, *args, **kwargs):
        self._unsupported_exact("amplitude")

    def sample(self, *args, **kwargs):
        self._unsupported_exact("sample")

    def sample_chaotic(self, *args, **kwargs):
        self._unsupported_exact("sample_chaotic")

    def sample_chaotic_rehearse(self, *args, **kwargs):
        self._unsupported_exact("sample_chaotic_rehearse")

    def partial_trace(self, *args, **kwargs):
        self._unsupported_exact("partial_trace")

    @property
    def uni(self):
        raise NotImplementedError(
            "`uni` (the dense circuit unitary) is not available for "
            "`CircuitPEPSSimpleUpdate`, which never forms the full unitary. "
            "Apply gates to a state and use `psi` or `local_expectation`."
        )

    def get_psi_reverse_lightcone(self, where, keep_psi0=False):
        # the reverse lightcone is not meaningful for a simple update PEPS,
        # which always keeps the whole state, so just return it
        return self.psi
