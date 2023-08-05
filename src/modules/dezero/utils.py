"""
utils
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import MutableSequence

from src.modules.dezero import Variable, Function
from src.modules.lower_layer_modules.FileSideEffects import prepare_directory


def _dot_var(v: Variable, verbose: bool = False) -> str:
    """
    _dot_var
    """
    dot_var: str = "{} [label=\"{}\", color=orange, style=filled]\n"
    name: str = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)
    return dot_var.format(id(v), name)


def _dot_func(f: Function) -> str:
    """
    _dot_func
    """
    dot_func: str = "{} [label=\"{}\", color=lightblue, style=filled, shape=box]\n"
    txt: str = dot_func.format(id(f), f.__class__.__name__)

    dot_edge: str = "{} -> {}\n"
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))
    return txt


def get_dot_graph(output: Variable, verbose: bool = True) -> str:
    """
    get_dot_graph
    """
    txt: str = ""
    funcs: MutableSequence[Function] = []
    seen_set: set[Function] = set()

    def add_func(f: Function) -> None:
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func: Function = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)
    return "digraph g {\n" + txt + "}"


def plot_dot_graph(output: Variable, verbose: bool = True, to_file: str | None = None) -> None:
    """
    plot_dot_graph
    """
    dot_graph: str = get_dot_graph(output, verbose)
    tmp_dir: Path = Path(os.path.expanduser("~")) / ".dezero"
    prepare_directory(tmp_dir)
    graph_path: Path = tmp_dir / "tmp_graph.dot"
    with open(graph_path, "w") as f:
        f.write(dot_graph)
    extension: str = "png" if to_file is None else Path(to_file).suffix[1:]
    command: str = f"dot {str(graph_path)} -T {extension} -o {to_file}"
    subprocess.run(command, shell=True)
