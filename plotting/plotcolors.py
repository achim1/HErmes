from pyevsel.utils.logger import Logger
import os.path

STD_CONF=os.path.join(os.path.split(__file__)[0],"plotsconfig.yaml")


seaborn_loaded = False
try:
    import seaborn.apionly as sb

    seaborn_loaded = True
    Logger.info("Seaborn found!")
except ImportError:
    Logger.warning("Seaborn not found! Using predefined color palette")

if not seaborn_loaded:
    color_palette = dict()   # stolen from seaborn color-palette
    color_palette[0]           = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)
    color_palette[-1]     = (0.8, 0.7254901960784313, 0.4549019607843137)#(0.3921568627450    9803, 0.7098039215686275, 0.803921568627451)
    color_palette["k"]           = "k"
    color_palette[1]       = (0.3333333333333333, 0.6588235294117647, 0.40784313725490196)
    color_palette[2] = (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)
    color_palette[3]    = (0.5058823529411764, 0.4470588235294118, 0.6980392156862745)

else:
    color_palette = sb.color_palette("dark")


############################################

def GetColorPalette(name="dark"):
    """
    Load a color pallete, use seaborn if available
    """

    if not seaborn_loaded:
        return color_palette
    else:
        return sb.color_palette(name)        


