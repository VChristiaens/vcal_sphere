Several example parameter files are provided in this folder.

- Calibration: the observation bands need to be adapted. Note that this is left to the user, as some targets were observed in different bands over the same night, which may involve several bands being present in the downloaded data => in such case the user will have to run the pipeline several times separately, once per band.

- Preprocessing: plate scale parameters need to be adapted depending on the band (see Maire et al. 2016). The "true_ncen" parameter needs to be adapted based on the number of CENTER calibration files available -- more exactly the number of "timesteps" probed by the CENTER files (i.e. 1=just beginning or just end; 2=beginning+end, 3=beginning+middle+end). Sometimes several CENTER files are taken consecutively but probe the same "timestep".

- Postproc: will mostly depend on the strategy to be adopted (ADI or RDI).

- Note that reference star datasets to be used for RDI require running separately the calibration and preprocessing stages, similarly to the science target. Only the postproc parameter file for the science target then needs to be adapted to provide the reference cube filename.