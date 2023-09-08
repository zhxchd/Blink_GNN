from data.citeseer import load_citeseer
from data.cora import load_cora
from data.facebook import load_facebook
from data.lastfm import load_lastfm
from data.pubmed import load_pubmed
from data.twitch import load_twitch

def make_dataset(dataset_name, root):
    if dataset_name == "cora":
        return load_cora(root)
    elif dataset_name == "pubmed":
        return load_pubmed(root)
    elif dataset_name == "lastfm":
        return load_lastfm(root)
    elif dataset_name == "citeseer":
        return load_citeseer(root)
    elif dataset_name == "facebook":
        return load_facebook(root)
    elif dataset_name == "twitch":
        return load_twitch(root)