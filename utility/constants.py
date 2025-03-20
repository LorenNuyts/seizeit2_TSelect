SEED = 0

class Nodes:
    # eeg_nodes = ['Fp1', 'Fp2', 'AF11', 'AF12', 'AF1', 'AF2', 'AF7', 'AF8', 'B2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'P7',
    #              'P3', 'Pz', 'P4', 'P8', 'O1', 'O2', 'F9', 'F10', 'T9', 'T10', 'P9', 'P10', 'FC5', 'FC1', 'FC2', 'FC6',
    #              'CP5', 'CP1', 'CP2', 'CP6', 'PO1', 'PO2', 'IZ',  'T7', 'C3', 'Cz', 'C4', 'T8', 'A1', 'A2', 'B1', 'F7',
    #              'FT9', 'FT10', 'FT7', 'FT8', 'TP7', 'TP8', 'B2']
    # wearable_nodes = ['BTEleft SD', 'BTEright SD', 'CROSStop SD',]
    #
    # eeg_acc = ['EEG SD ACC X', 'EEG SD ACC Y', 'EEG SD ACC Z']
    # eeg_gyr = ['EEG SD GYR A', 'EEG SD GYR B', 'EEG SD GYR C']
    # ecg_emg_nodes = ['ECG+', 'sEMG+', 'ECG SD', 'EMG SD', 'ECGEMG SD ACC X', 'ECGEMG SD ACC Y', 'ECGEMG SD ACC Z',
    #                  'ECGEMG SD GYR A', 'ECGEMG SD GYR B', 'ECGEMG SD GYR C']
    # other_nodes = ['Ment+', 'A1*', 'A2*', 'MKR+', 'Iz']

    # eeg_nodes = ['Fp1', 'Fp2', 'AF11', 'AF12', 'AF1', 'AF2', 'AF7', 'AF8', 'B2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'P7',
    #              'P3', 'Pz', 'P4', 'P8', 'O1', 'O2', 'F9', 'F10', 'T9', 'T10', 'P9', 'P10', 'FC5', 'FC1', 'FC2', 'FC6',
    #              'CP5', 'CP1', 'CP2', 'CP6', 'PO1', 'PO2', 'Iz', 'T7', 'C3', 'Cz', 'C4', 'T8', 'A1', 'A2', 'FT9',
    #              'FT10',
    #              'FT7', 'FT8', 'TP7', 'TP8']

    basic_eeg_nodes = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2']
    optional_eeg_nodes = ['AF11', 'AF12', 'AF1', 'AF2', 'AF7', 'AF8', 'B2', 'F7', 'F8', 'P7', 'P8', 'F9', 'F10', 'T9',
                          'T10',
                          'P9', 'P10', 'FC5', 'FC1', 'FC2', 'FC6', 'CP5', 'CP1', 'CP2', 'CP6', 'PO1', 'PO2', 'Iz', 'T7',
                          'T8',
                          'A1', 'A2', 'FT9', 'FT10', 'FT7', 'FT8', 'TP7', 'TP8']

    wearable_nodes = ['BTEleft SD', 'BTEright SD', 'CROSStop SD']
    BTEleft = 'BTEleft SD'
    BTEright = 'BTEright SD'
    CROSStop = 'CROSStop SD'

    eeg_acc = ['EEG SD ACC X', 'EEG SD ACC Y', 'EEG SD ACC Z']
    eeg_gyr = ['EEG SD GYR A', 'EEG SD GYR B', 'EEG SD GYR C']

    ecg_emg_nodes = ['ECG+', 'sEMG+', 'ECG SD', 'EMG SD']
    ecg_emg_acc = ['ECGEMG SD ACC X', 'ECGEMG SD ACC Y', 'ECGEMG SD ACC Z']
    ecg_emg_gyr = ['ECGEMG SD GYR A', 'ECGEMG SD GYR B', 'ECGEMG SD GYR C']

    other_nodes = ['Ment+', 'A1*', 'A2*', 'MKR+', 'B1', 'B2']

    switchable_nodes = {
        BTEright: [BTEleft, CROSStop],
        BTEleft: [BTEright, CROSStop],
        CROSStop: [BTEright, BTEleft]
    }


class Locations:
    coimbra = "Coimbra_University_Hospital"
    freiburg = "Freiburg_University_Medical_Center"
    karolinska = "Karolinska_Institute"
    leuven_adult = "University_Hospital_Leuven_Adult"
    leuven_pediatric = "University_Hospital_Leuven_Pediatric"
    aachen = "University_of_Aachen"

class Keys:
    pass