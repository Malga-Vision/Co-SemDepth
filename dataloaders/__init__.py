from .midair import DataLoaderMidAir as MidAir
#from .midair_origin import DataLoaderMidAir as MidAir
#from .kitti import DataLoaderKittiRaw as KittiRaw
from .tartanair import DataLoaderTartanAir as TartanAir
from .wilduav import DataLoaderWUAV as WildUAV
from .uzh import DataLoaderUZH as UZH
from .topair import DataLoaderTopAir as TopAir
from .cityscapes import DataLoaderCityScapes as CityScapes
from .generic import DataloaderParameters

def get_loader(name : str):
    available = {
        "midair"        : MidAir(),
        #"kitti-raw"     : KittiRaw(),
        "tartanair"   : TartanAir(),
        "wilduav"   : WildUAV(),
        "uzh"   : UZH(),
        "topair" : TopAir(),
        "cityscapes": CityScapes()
    }
    try:
        return available[name]
    except:
        print("Dataloaders available:")
        print(available.keys())
        raise NotImplementedError
