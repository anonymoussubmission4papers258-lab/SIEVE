"""SIEVE-3: correspondence-coupled super-resolution for hyperspectral-RGB fusion.

The sub-block spectral detail an LR-HSI pixel averages away and RGB cannot see (null(A) x null(S)) is not
guessed: it is *measured*, by coupling every pixel to the other places in the scene where the same point of
the same kind of thing was imaged at a different sub-block phase of the LR grid.

* the VFM (sieve3.model.CoupledNet) reads RGB + LR-HSI jointly and outputs couplings only:
  material slots Pi, part codes Gamma, coupling gate beta, correction gate g -- never a spectrum;
* the measurements solve every spectral value in closed form (sieve3.coupled);
* held-out spectral evidence trains the couplings, gates the correction and audits the couplings
  (sieve3.evidence, sieve3.audit): copies that look alike but measure differently are exposed and decoupled;
* the measured map mu says, per pixel, how much of the coupled detail the measurements pin down.
"""
__version__ = "0.3.0"
