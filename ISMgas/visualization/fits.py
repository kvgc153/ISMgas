import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize, LinearStretch, LogStretch, PowerStretch, SqrtStretch, PercentileInterval
from regions import Regions

def regionToMask(filename, data, kwargs={'format':'ds9'}):
    """Generates a mask from a region file.

    Args:
        filename (str): Region file name.
        data (np.array): Data array.
        kwargs (dict, optional): _description_. Defaults to {'format':'ds9'}.

    Returns:
        np.array: Mask array.
    """
    regions = Regions.read(filename, **kwargs)
    mask = regions[0].to_mask()
    return mask.to_image(data.shape)

class ScaleImage:
    def __init__(self, image:'numpy.array', scale:'str'= 'percentile', scale_kwargs:dict = {'percentile':95}, cmap:str = 'gray'):
        """This class is used to scale and plot astronomical images.

        Args:
            image (numpy.array): _description_
            scale (str, optional): _description_. Defaults to 'percentile'.
            scale_kwargs (_type_, optional): _description_. Defaults to {'percentile':95}.
            cmap (str, optional): _description_. Defaults to 'gray'.
        """
        
        self.image = image

        scalesDB ={
            'percentile': PercentileInterval,
            'zscale'    : ZScaleInterval,

        }
        self.scale = scalesDB[scale]
        self.scale_kwargs = scale_kwargs


        stretchDB = {
            'linear': LinearStretch,
            'log'   : LogStretch,
            'power' : PowerStretch,
            'sqrt'  : SqrtStretch, 
        }
        # self.stretch = stretchDB[stretch]

        self.cmap = cmap
        
    def plot(self, origin='lower'):
        plt.imshow(
            self.image, 
            cmap=self.cmap,
            norm=ImageNormalize(
                self.image, 
                interval=self.scale(**self.scale_kwargs)
                ),
            origin=origin  
        )