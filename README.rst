vcal_sphere
-----------
Pipeline for calibration, pre-processing and post-processing of SPHERE IRDIS and IFS data, based on VIP routines and esorex recipes.


Installation
------------
For now: only manually by git fork, and adding the path for the main vcal_sphere modules to your $PYTHONPATH.


Requirements
------------
- VIP: https://github.com/vortex-exoplanet/VIP
- esorex: https://www.eso.org/sci/software/cpl/esorex.html


Checklist
---------
Before running the pipeline, make sure:

1) you only keep in your 'raw' folder raw flats and darks from the same morning! I.e. remove all the other ones in different folders!
2) you downloaded manually raw IFS darks with matching DITs to the FLATS - they don't come automatically just by ticking the box of downloading data with raw calibs!
3) first try basic calibration using default parameters - change them only if you know what you're doing.


Acknowledgements
----------------
If you use `vcal_sphere`, please cite `Christiaens et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.6117C/abstract>`_. 