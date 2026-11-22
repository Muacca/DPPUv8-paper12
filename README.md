# Adiabatic Endpoint Polarizations for Dirac Fields on Anisotropic Compact Cosmologies: Fock Implementability and Excitation Energy

Muacca · DPPUv8 Paper 12 · **Version 1** · First release 2026-11-22

The spectrum at two round ends of a biaxial sphere agrees, but the changing shape axis mixes particles and holes. For the stated massless Dirac field we prove when total number and occupation-weighted energy are finite, bound the finite quantities, and compare the Fock criterion on compact Nil.

[English manuscript (PDF)](DPPUv8-paper12.pdf) · [LaTeX source](LaTeX/main.tex) · [日本語本文・目次](README_ja.md) · [References](REFERENCES.md) · [Changelog](CHANGELOG.md)

## Reproduce

~~~sh
python -m pip install -r script/requirements.txt
python -B script/paper12.py --figure figures_new/history.pdf
python -B script/sphere_certificate.py --output results_new/sphere.json
python -B script/nil_transport.py --output results_new/nil_paired.json
python -B script/third_order.py --output results_new/third_order.json
~~~

[Data and conventions](data/README.md) · [Code and commands](script/README.md) · [Build instructions](LaTeX/README.md) · [MIT license](LICENSE.txt)

## Acknowledgments

The author acknowledges Claude Opus 5.5, Gemini 4 Neon and GPT Astra 6.2 for assistance during preparation of the manuscript in November 2026, including symbolic derivations, implementation, Japanese–English exposition and deterministic figure-generation code. The calculations and figures are reproducible from the supplied mathematical definitions and data. The author takes responsibility for the content and remaining errors.

This work was developed within the informal collaborative project **DPPU** (**D**onut-like topology, **P**lanck-scale compactness, **P**recession dynamics, **U**niverse).
