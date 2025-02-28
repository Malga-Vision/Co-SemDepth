from .midair import DataLoaderMidAir as MidAir
from .topair import DataLoaderTopAir as TopAir
from .generic import DataloaderParameters

def get_loader(name : str):
    available = {
        "midair"        : MidAir(),
        "topair" : TopAir()
    }
    try:
        return available[name]
    except:
        print("Dataloaders available:")
        print(available.keys())
        raise NotImplementedError
