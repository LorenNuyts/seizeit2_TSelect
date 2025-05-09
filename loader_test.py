from importlib import resources as impresources
from pathlib import Path
from data.data import Data
from data.annotation import Annotation

# data_path = Path('/esat/biomeddata/SeizeIT2/bids')       # path to dataset
data_path = Path("/media/loren/Seagate Basic/Epilepsy use case")       # path to dataset
hospital = 'Coimbra_University_Hospital'
data_path = data_path / hospital
## Build recordings list:
sub_list = [x for x in data_path.glob("SUB*")]
# recordings = [[x.name, xx.name.split('_')[-2]] for x in sub_list for xx in (x / 'ses-01' / 'eeg').glob("*edf")]

# filter recordings to choose only recordings from certain patient:
recordings = [x for x in sub_list if 'SUBJ-7-331' in x.name.split('_')[-1]]

data = list()
annotations = list()

for rec in recordings:
    print(rec[0] + ' ' + rec[1])
    rec_data = Data.loadData(data_path.as_posix(), rec, modalities=['eeg', 'ecg', 'mov'])
    rec_annotations = Annotation.loadAnnotation(data_path.as_posix(), rec)

    data.append(rec_data)
    annotations.append(rec_annotations)

