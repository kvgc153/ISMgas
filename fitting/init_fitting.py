def priorsInit():
    priors = {}

    ### constraint_option ==1
    priors["1"] ={
        "Amin"        : 0.1,
        "Amax"        : 1.0,
        "A1min"       : 0.1,
        "A1max"       : 1.0,
        "Amin_SG"     : 0.1,
        "Amax_SG"     : 1.0,
        "vmin"        : -700,
        "vmax"        : 500,
        "v1min"       : -700,
        "v1max"       : 500,
        "vmin_SG"     : -700,
        "vmax_SG"     : 500,
        "sigmin"      : 50,
        "sigmax"      : 700,
        "sig1min"     : 50,
        "sig1max"     : 700,
        "sigmin_SG"   : 50,
        "sigmax_SG"   : 700,
        "v-v1_min"    : 0, 
        "v-v1_max"    : -800,
        "cont_lvl"    : 1 ### Choose continuum level
    }
    return(priors)
