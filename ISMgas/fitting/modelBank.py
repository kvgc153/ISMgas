from astropy.modeling import models, fitting
from ISMgas.linelist import linelist_highz

G1D       = models.Gaussian1D
L1D       = models.Linear1D
P1D       = models.Polynomial1D
C1D       = models.Const1D

# def buildLine(lineName:str, **kwargs):
#     """Build a line model from a dictionary of parameters"""
#     name = lineName
#     wavelength = linelist_highz[lineName]['lambda']

# def create_G1D(amplitude, mean, stddev, bounds_amplitude, bounds_mean, bounds_stddev):
#     return G1D(
#         amplitude=amplitude, 
#         mean=mean,  
#         stddev=stddev,  
#         bounds={
#             'amplitude': bounds_amplitude,
#             'mean': bounds_mean,
#             'stddev': bounds_stddev
#         }    
#     )

SiIII_1294 = G1D(
    amplitude=0.2, 
    mean= linelist_highz['Si III 1294']['lambda'],  
    stddev=0.4,  
    bounds={
        'amplitude':[0,1],
        'mean':[1294,1295],
        'stddev':[0.4,0.7]
        
    }    
)

interveningII = G1D(
    amplitude=0.2,
    mean=1296.33, 
    stddev=0.4,
    bounds={
        'amplitude':[0,1],
        'mean':[1296,1296.5],
        'stddev':[0.4,0.7]
        
    }    
)

SiIII_1298 = G1D(
    amplitude=0.2, 
    mean= linelist_highz['Si III 1298-8']['lambda'], 
    stddev= 0.4,
    bounds={
        'amplitude':[0,1],
        'mean':[1298,1299],
        'stddev':[0.4,0.7]
        
    }    
)
OI_1302   = G1D(
    amplitude=0.2,
    mean= linelist_highz['O I 1302']['lambda'],  
    stddev=0.4,
    bounds={
       'amplitude':[0.1,1],
       'mean':[1302,1303],
       'stddev':[0.3,0.7]
    }    
)

SiII_1304 = G1D(
    amplitude=0.2, 
    mean= linelist_highz['Si II 1304']['lambda'], 
    stddev=0.4,
    bounds={
       'amplitude':[0.1,1],
       'mean':[1303,1305],
       'stddev':[0.3,0.7]
    }    
)


CIII_1247 =  G1D(
    name = "C III 1247",
    amplitude=0.2, 
    mean=linelist_highz['C III 1247']['lambda'], 
    stddev=0.4,
    bounds={
        'amplitude':[0,1],
        'mean':[1246,1248],
        'stddev':[0.4,0.7]

    }
)


SII_1250 = G1D(
    amplitude=0.2, 
    mean=linelist_highz['S II 1250']['lambda'], 
    stddev=0.4,
    bounds={
        'amplitude':[0,1],
        'mean':[1249,1251],
        'stddev':[0.4,0.7]
        
    }
)

SII_1253 = G1D(
    amplitude=0.2, 
    mean=linelist_highz['S II 1253']['lambda'], 
    stddev=0.4,
    bounds={
        'amplitude':[0,1],
        'mean':[1252,1254],
        'stddev':[0.4,0.7]
        
    }
)

SiII_1260 = G1D(
    amplitude=0.8, 
    mean=linelist_highz['Si II 1260']['lambda'], 
    stddev=0.4,
    bounds={
        'amplitude':[0.5,1],
        'mean':[1258,1261],
    }                          
) 

interveningI = G1D(
    amplitude=0.3, 
    mean=1264.5, 
    stddev=0.4,
    bounds={
        'amplitude':[0.2,1],
        'mean':[1264,1265],
    }                          
)


SiII_1526 = G1D(
    name = "Si II 1526",
    amplitude=0.2, 
    mean=linelist_highz['Si II 1526']['lambda'], 
    stddev=0.4,
    bounds={
       'amplitude':[0.1,1],
       'mean':[1526,1527.5],
        'stddev':[0,1]
    }
)

SiII_1526_outflow = G1D(
    name = "Si II 1526 - outflow",
    amplitude=0.3, 
    mean=linelist_highz['Si II 1526']['lambda']-1, 
    stddev=0.4,
    bounds={
       'amplitude':[0,1],
       'mean':[1525,1527],
        'stddev':[0,1]

    }
) 



CIV_1548_broad = G1D(
    name = "CIV - broad",
    amplitude=0.2, 
    mean=linelist_highz['C IV  1548']['lambda'], 
    stddev=0.4,
    bounds={
       'amplitude':[0,5],
       'mean':[1527,1548],
       'stddev':[0,15]
    }
)

CIV_1548 = G1D(
    name = "C IV 1548",
    amplitude=1, 
    mean=linelist_highz['C IV  1548']['lambda'], 
    stddev=0.4,
    bounds={
       'amplitude':[0.2,2],
       'mean':[1547.5,1549],
        'stddev':[0,1]
    }
)

CIV_1550 = G1D(
    name = "C IV 1550",
    amplitude=1, 
    mean=linelist_highz['C IV  1550']['lambda'], 
    stddev=0.4,
    bounds={
       'amplitude':[0.2,2],
       'mean':[1549.5,1551],
        'stddev':[0,1]
    }
)


SiII_FE = G1D(
    name = "SiII FE 1534",
    amplitude=0.05, 
    mean=linelist_highz['Si II* 1533']['lambda'], 
    stddev=0.4,
    bounds={
        'amplitude':[0,5],
        'mean':[1533,1535],
        'stddev':[0,1.2]
        
    }
)

HeII_1640_narrow = G1D(
    name = "He II 1640 - narrow",
    amplitude=1, 
    mean=linelist_highz['He II 1640']['lambda'], 
    stddev=0.4,
    bounds={
       'amplitude':[0.2,2],
       'mean':[1640,1641],
        'stddev':[0,1]
    }
)

HeII_1640_broad = G1D(
    name = "He II 1640 - broad",
    amplitude=0.2, 
    mean=linelist_highz['He II 1640']['lambda'], 
    stddev=0.4,
    bounds={
       'amplitude':[0,5],
       'mean':[1637,1642],
       'stddev':[1,10]
    }
)

AlII_1670 = G1D(
    name = "Al II 1670",
    amplitude=0.2, 
    mean=linelist_highz['Al II 1670']['lambda'], 
    stddev=0.4,
    bounds={
       'amplitude':[0.1,1],
       'mean':[1670,1672],
        'stddev':[0,1]
    }
)

AlII_1670_outflow = G1D(
    name = "Al II 1670 - outflow",
    amplitude=0.3, 
    mean=linelist_highz['Al II 1670']['lambda']-1, 
    stddev=0.4,
    bounds={
       'amplitude':[0,1],
       'mean':[1669,1671],
        'stddev':[0,1]

    }
) 

OIII_1661 = G1D(
    name = "O III] 1661",
    amplitude=0.05, 
    mean=linelist_highz['O III] 1661']['lambda'], 
    stddev=0.2,
    bounds={
       'amplitude':[0.05,1],
       'mean':[1659.8,1661.5],
        'stddev':[0,1]

    }
)


OIII_1661_outflow = G1D(
    name = "O III] 1661 - outflow",
    amplitude=0.05, 
    mean=linelist_highz['O III] 1661']['lambda'], 
    stddev=0.4,
    bounds={
       'amplitude':[0,1],
       'mean':[1659,1661],
       'stddev':[0.,1.5]

    }
)

OIII_1666 = G1D(
    name = "O III] 1666",
    amplitude=0.2, 
    mean=linelist_highz['O III] 1666']['lambda'], 
    stddev=0.2,
    bounds={
       'amplitude':[0.1,1],
       'mean':[1665,1667],
        'stddev':[0,1]

    }    
)

OIII_1666_outflow = G1D(
    name = "O III] 1666 - outflow",
    amplitude=0.05, 
    mean=linelist_highz['O III] 1666']['lambda'], 
    stddev=0.7,
    bounds={
       'amplitude':[0.,1],
       'mean':[1665,1666.5],
       'stddev':[0.,1.5]

    }    

)

FeII_1608 = G1D(
    name = "Fe II 1608",    
    amplitude=0.2, 
    mean=linelist_highz['Fe II 1608']['lambda'], 
    stddev=0.4,
    bounds={
       'amplitude':[0.,1],
       'mean':[1607,1609],
       'stddev':[0,0.7]
    }
)

FeII_1611 = G1D(
    name = "Fe II 1611",
    amplitude=0.2, 
    mean=linelist_highz['Fe II 1611']['lambda'], 
    stddev=0.4,
    bounds={
       'amplitude':[0.,1],
       'mean':[1610,1612],
       'stddev':[0,0.7]
    }
)

SiII_1808 = G1D(
    name = "Si II 1808",
    amplitude=0.2, 
    mean=linelist_highz['Si II 1808']['lambda'], 
    stddev=0.4,
    bounds={
       'amplitude':[0.,1],
       'mean':[1807,1809],
       'stddev':[0,0.7]
    }
)

AlIII_1854 = G1D(
    name = "Al III 1854",
    amplitude=0.05, 
    mean=linelist_highz['Al III 1854']['lambda'], 
    stddev=0.4,
    bounds={
       'amplitude':[0,5],
       'mean':[1853,1855],
       'stddev':[0,1.2]

})
                     
AlIII_1862 = G1D(
    name = "Al III 1862",
    amplitude=0.05, 
    mean=linelist_highz['Al III 1862']['lambda'], 
    stddev=0.4,
    bounds={
        'amplitude':[0,5],
        'mean':[1861,1863],
        'stddev':[0,1.2]
        
    }
)


SiIII_1882 = G1D(
    name = "Si III] 1882",
    amplitude=0.05, 
    mean=linelist_highz['Si III] 1882']['lambda'], 
    stddev=0.4,
    bounds={
        'amplitude':[0,5],
        'mean':[1881,1883],
        'stddev':[0,1.2]
        
    }
)

SiIII_1892 = G1D(
    name = "Si III] 1892",
    amplitude=0.05, 
    mean=linelist_highz['Si III] 1892']['lambda'], 
    stddev=0.4,
    bounds={
        'amplitude':[0,5],
        'mean':[1891,1893],
        'stddev':[0,1.2]
        
    }
)


CIII_1907  =  G1D(
    name = "CIII] 1907", 
    amplitude=0.2, 
    mean=linelist_highz['C III] 1907']['lambda'], 
    stddev=0.4,
    bounds={
        'amplitude':[0,5],
        'mean':[1905.5,1908],
        'stddev':[0,1]

    }
)    


CIII_1907_outflow  =  G1D(
    name = "CIII] 1907 - outflow",
    amplitude=0.05, 
    mean=linelist_highz['C III] 1907']['lambda'], 
    stddev=0.4,
    bounds={
        'amplitude':[0,5],
        'mean':[1905,1907],
        'stddev':[0,1.2]
        
    }
)

CIII_1909 =   G1D(
    name = "CIII] 1909", 
    amplitude=0.2, 
    mean=linelist_highz['C III] 1909']['lambda'], 
    stddev=0.4,
    bounds={
        'amplitude':[0,5],
        'mean':[1908,1909.3],
        'stddev':[0,1]
        
    }
) 

CIII_1909_outflow = G1D(
    name   = "CIII] 1909 - outflow",
    amplitude=0.05, 
    mean=linelist_highz['C III] 1909']['lambda'], 
    stddev=0.4,
    bounds={
       'amplitude':[0,5],
       'mean':[1907,1909],
       'stddev':[0,1.2]

    }
)
