vcal_sphere
-----------
Pipeline for calibration, pre-processing and post-processing of SPHERE IRDIS and IFS data, based on VIP routines and esorex recipes.

.. image:: https://github.com/user-attachments/assets/16198baa-e739-43c4-9d98-ca4410c5eeee
   :width: 400

Installation
------------
For now, only through Git: either clone it directly, or fork it first (the latter if you plan to contribute/debug).

If you first fork the repository, click on the fork button in the top right corner.
Then clone it:

.. code-block:: bash

  $ git clone https://github.com/<replace-by-your-username>/vcal_sphere.git

If you do not create a fork, you can still benefit from the ``git`` syncing
functionalities by cloning the repository (but will not be able to contribute):

.. code-block:: bash

  $ git clone https://github.com/VChristiaens/vcal_sphere.git

Before installing the package, it is highly recommended to create a dedicated
conda environment to not mess up with the package versions in your base 
environment. This can be done easily with (replace vcal_env by the name you want
for your environment):

.. code-block:: bash

  $ conda create -n vcal_env python=3.10 ipython

Note: installing ipython while creating the environment with the above line will
avoid a commonly reported issue which stems from trying to import VIP from 
within a base python2.7 ipython console.

To install vcal, simply cd into the vcal_sphere directory and run the setup file 
in 'develop' mode:

.. code-block:: bash

  $ cd vcal_sphere
  $ python setup.py develop

If cloned from your fork, make sure to link your vcal_sphere directory to the upstream 
source, to be able to easily update your local copy when a new version comes 
out or a bug is fixed:

.. code-block:: bash

  $ git add remote upstream https://github.com/VChristiaens/vcal_sphere.git


Requirements
------------
- VIP: https://github.com/vortex-exoplanet/VIP
- ESO SPHERE pipeline kit: The source kit (with esorex and all prerequisites bundled) can be downloaded from ESO or installed through MacPorts (https://www.eso.org/sci/software/pipelines/). A more detailed procedure for the SPHERE kit is given here: https://www.eso.org/sci/software/pipelines/sphere/sphere-pipe-recipes.html


Quick start
-----------
Follow the instructions in the VCAL_QUICKSTART.txt file.


Checklist
---------
Before running the pipeline, make sure:

1) you only keep in your 'raw' folder raw flats and darks from the same morning! I.e. remove all the other ones in different folders.
2) If planning to reduce IFS data: make sure raw IFS darks with matching DITs to the FLATS are included among your raw data. If not you will have to download them manually - as of 2023 they don't come automatically just by ticking the box of downloading the data with raw associated calibrations.
3) First try basic calibration and preprocessing using default parameters - change them only if the master cubes obtained by the end of preprocessing look poor. 


Contact
-------
Feel free to raise any issue on the GitHub. 
If you have questions on proper usage of the code, assess whether the data look well calibrated/pre-processed/post-processed. If you would like to contribute to the code, feel free to contact me at valentin.christiaens@uliege.be.


Acknowledgements
----------------
If `vcal_sphere` is useful to your work, please cite `Christiaens et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023ascl.soft11002C/abstract>`_.
