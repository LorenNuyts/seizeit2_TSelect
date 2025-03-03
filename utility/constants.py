class Nodes:
    eeg_nodes = ['Fp1', 'Fp2' 'AF11', 'AF12', 'AF1', 'AF2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'P7', 'P3', 'Pz', 'P4', 'P8',
                 'O1', 'O2', 'F9', 'F10', 'T9', 'T10', 'P9', 'P10', 'FC5', 'FC1', 'FC2', 'FC6', 'CP5', 'CP1', 'CP2',
                 'CP6', 'PO1', 'PO2', 'IZ',  'T7', 'C3', 'Cz', 'C4', 'T8']
    wearable_nodes = ['BTEleft SD', 'BTEright SD', 'CROSStop SD',]

    eeg_acc = ['EEG SD ACC X', 'EEG SD ACC Y', 'EEG SD ACC Z']
    eeg_gyr = ['EEG SD GYR A', 'EEG SD GYR B', 'EEG SD GYR C']
    ecg_emg_nodes = ['ECG+', 'SEMG+', 'ECG SD', 'EMG SD', 'ECGEMG SD ACC X', 'ECGEMG SD ACC Y', 'ECGEMG SD ACC Z',
                     'ECGEMG SD GYR A', 'ECGEMG SD GYR B', 'ECGEMG SD GYR C']
    other_nodes = ['Ment+', 'B1', 'A1*', 'A2*', 'MKR+', ]

class Locations:
    coimbra = "Coimbra_University_Hospital"
    freiburg = "Freiburg_University_Medical_Center"
    karolinska = "Karolinska_Institute"
    leuven_adult = "University_Hospital_Leuven_Adult"
    leuven_pediatric = "University_Hospital_Leuven_Pediatric"
    aachen = "University_of_Aachen"

class Keys:
    pass