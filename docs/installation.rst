Installation
============

Install the released package with::

    python -m pip install spikes-rs

Python needs version 3.12 or later and NumPy 2.0 or later. To build from a checkout instead,
which needs Rust 1.99 or later, run from the project root::

    python -m pip install .

Development
-----------

Install the test and documentation tools and build the extension into the active environment::

    python -m pip install maturin
    maturin develop --release --extras dev
    cargo test
    pytest --pyargs spikes_rs
    sphinx-build -W docs docs/_build/html
    prek run --all-files

Rebuild the extension after editing Rust. Run ``tox -e py314-cparity`` with Python 3.14
to install the pinned ``ftoolss`` reference and compare the spike masks pixel by pixel.

Time one saved exposure with ``python benchmarks/throughput.py exposure.npy``.
The reported median includes preparing the clipping buffer and allocating the mask.
