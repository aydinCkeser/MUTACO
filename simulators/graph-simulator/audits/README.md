# Historical graph fixture producer

[capture_retained_graph_fixture.py](capture_retained_graph_fixture.py) runs a
small fixed graph problem in a supplied historical checkout using its NumPy
and Numba implementation. It records the source Git revision, source hashes,
kernel options, tensor axes, dtypes, data bytes and per-tensor hashes at an
explicit new output path. The script refuses to overwrite an existing file.

This is a **physical reference producer**. Viewing its `--help` prints argument
information; invoking it with `--historical-repository` and `--output` can
execute the historical simulator. Neither ordinary imports nor saved-result
verification perform that computation. The separately
installed `mutaco_graph` package does not install this research script.
