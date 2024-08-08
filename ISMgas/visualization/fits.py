import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize, LinearStretch, LogStretch, PowerStretch, SqrtStretch, PercentileInterval
from regions import Regions
import matplotlib.pyplot as plt

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


def mockData2D():
    """Generates a 2D mock image."""
    # Parameters
    size = 500  # Size of the grid
    radius = 5
    frequency = 5

    # Create a grid of coordinates
    x = np.linspace(-radius, radius, size)
    y = np.linspace(-radius, radius, size)
    X, Y = np.meshgrid(x, y)

    # Calculate the distance from the center
    R = np.sqrt(X**2 + Y**2)

    # Create the circular sine wave pattern
    Z = np.sin(frequency * R)/R

    # Plot using imshow
    plt.figure(figsize=(8, 8))
    plt.imshow(Z, extent=(-radius, radius, -radius, radius), origin='lower', cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.title('Mock 2D image ')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
    return Z


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