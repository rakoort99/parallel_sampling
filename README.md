# parallel_sampling

Code for a paper on parallel sampling methods. Implement sequential Gibbs, Hogwild Gibbs (De Saa et al. 2017), asynchronous exact Gibbs* (Terenin et al. 2020), distributed Metropolis* (Feng et al. 2019), Local Glauber Dynamics* (Fischer and Ghaffari 2018), and Chromatic Gibbs* (Gonzalez et al. 2011). 

Functions for each sampling method can be imported from `samplers.py`. To replicate figures, run `make_plots.py`.

Due to the limitations of Python and for the sake of productivity, I implement exact sequential replicas of algorithms with an asterisk next to their name.
