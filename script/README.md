# Reproducibility

Run these commands from the repository root with Python 3.11 or later and the libraries in [requirements.txt](requirements.txt). Tested with Python 3.12.10, NumPy 2.3.5, SciPy 1.16.3, python-flint 0.8.0 and Matplotlib 3.10.7.

~~~sh
python -m pip install -r script/requirements.txt
python -B script/paper12.py --figure figures_new/history.pdf
python -B script/sphere_certificate.py --output results_new/sphere.json
python -B script/nil_transport.py --output results_new/nil_paired.json
python -B script/third_order.py --output results_new/third_order.json
~~~

The first checks the 551 allowed positive magnetic pairs and their weights, the published intervals after summing the tail, and the order-three and Nil tables, then plots the spherical benchmark if requested (a .pdf or .png filename; the distributed LaTeX/figures/history.pdf and history.png are these outputs). The second builds interval Taylor expansions and bounds all 551 mode probabilities and the infinite tail at 180-bit precision using python-flint. The third integrates the selected Nil blocks in the unmoved basis with DOP853; their UV proof is analytical. The fourth is the independent higher-cutoff calculation for Appendix B. Comparisons should use the endpoints and rounding displayed in the paper. Execution time for the full sphere calculations is longer than for reading the saved tables.
