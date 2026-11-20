Installation
============

From the project root, build and install the Python extension with::

    python -m pip install .

Source builds need Rust 1.99 or later. Python needs version 3.12 or later and NumPy 2.0 or later.

Development
-----------

Install the test and documentation tools and build the extension into the active environment::

    python -m pip install maturin
    maturin develop --release --extras dev
    cargo test
    pytest --pyargs spikes_rs
    sphinx-build -W docs docs/_build/html
    prek run --all-files

Rebuild the extension after editing Rust. Set the environment variable
``RAYON_NUM_THREADS`` before starting Python to control the Rust worker count.

Time one saved exposure with ``python benchmarks/throughput.py exposure.npy``.
The reported median includes preparing the clipping buffer and allocating the mask.
