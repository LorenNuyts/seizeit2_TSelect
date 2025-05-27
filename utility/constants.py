from TSelect.tselect.tselect.utils.metrics import auroc_score
from net.utils import get_sens_FA_score

SEED = 0

# This variable is used to make sure there is at least one subject with seizures in the validation set.
subjects_with_seizures = {'SUBJ-1a-159', 'SUBJ-7-376', 'SUBJ-4-259', 'SUBJ-1a-358', 'SUBJ-1a-153', 'SUBJ-6-463',
                          'SUBJ-1a-006', 'SUBJ-1b-178', 'SUBJ-6-430', 'SUBJ-6-472', 'SUBJ-6-298', 'SUBJ-6-304',
                          'SUBJ-4-149', 'SUBJ-5-294', 'SUBJ-4-166', 'SUBJ-1b-307', 'SUBJ-4-145', 'SUBJ-1a-082',
                          'SUBJ-4-097', 'SUBJ-4-151', 'SUBJ-4-169', 'SUBJ-4-385', 'SUBJ-4-392', 'SUBJ-1b-223',
                          'SUBJ-5-255', 'SUBJ-4-265', 'SUBJ-1a-471', 'SUBJ-1b-222', 'SUBJ-1a-482', 'SUBJ-4-203',
                          'SUBJ-7-438', 'SUBJ-6-261', 'SUBJ-1b-315', 'SUBJ-4-346', 'SUBJ-1b-077', 'SUBJ-6-276',
                          'SUBJ-6-483', 'SUBJ-7-379', 'SUBJ-6-273', 'SUBJ-1a-489', 'SUBJ-6-216', 'SUBJ-4-305',
                          'SUBJ-4-230', 'SUBJ-4-466', 'SUBJ-6-256', 'SUBJ-7-449', 'SUBJ-6-275', 'SUBJ-4-139',
                          'SUBJ-6-311', 'SUBJ-6-393', 'SUBJ-1a-353', 'SUBJ-4-254', 'SUBJ-7-331', 'SUBJ-7-441'}


class Nodes:

    basic_eeg_nodes = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2']
    T_nodes = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']
    optional_eeg_nodes = ['AF11', 'AF12', 'AF1', 'AF2', 'AF7', 'AF8', 'B2', 'F7', 'F8', 'P7', 'P8', 'F9', 'F10',
                          'P9', 'P10', 'FC5', 'FC1', 'FC2', 'FC6', 'CP5', 'CP1', 'CP2', 'CP6', 'PO1', 'PO2', 'Iz',
                          'A1', 'A2', 'FT9', 'FT10', 'FT7', 'FT8', 'TP7', 'TP8'] + T_nodes
    eeg_left_top = 'EEG LiOorTop'
    eeg_right_top = 'EEG ReOorTop'
    eeg_left_behind = 'EEG LiOorAchter'
    eeg_right_behind = 'EEG ReOorAchter'
    eeg_ears = ['EEG ReOorAchter', 'EEG LiOorAchter', 'EEG ReOorTop', 'EEG LiOorTop']

    BTEleft = 'BTEleft SD'
    BTEright = 'BTEright SD'
    CROSStop = 'CROSStop SD'
    wearable_nodes = [BTEleft, BTEright, CROSStop]

    eeg_acc = ['EEG SD ACC X', 'EEG SD ACC Y', 'EEG SD ACC Z']
    eeg_gyr = ['EEG SD GYR A', 'EEG SD GYR B', 'EEG SD GYR C']

    ecg = 'ECG+'
    ecg_emg_nodes = [ecg, 'sEMG+', 'EMG Li Ar', 'EMG EMG Re', 'ECG SD', 'EMG SD']
    ecg_emg_acc = ['ECGEMG SD ACC X', 'ECGEMG SD ACC Y', 'ECGEMG SD ACC Z']
    ecg_emg_gyr = ['ECGEMG SD GYR A', 'ECGEMG SD GYR B', 'ECGEMG SD GYR C']

    other_nodes = ['Ment+', 'A1*', 'A2*', 'MKR+', 'B1', 'B2', 'Light Stimuli']

    switchable_nodes = {
        BTEright: [BTEleft, CROSStop],
        BTEleft: [BTEright, CROSStop],
        CROSStop: [BTEright, BTEleft]
    }

    @staticmethod
    def format_node_name(node:str):
        if "EEG".lower() in node.lower():
            if "LiOorAcht".lower() in node:
                return Nodes.eeg_left_behind
            elif "LiOorTop".lower() in node:
                return Nodes.eeg_left_top
            elif "ReOorAcht".lower() in node:
                return Nodes.eeg_right_behind
            elif "ReOorTop".lower() in node:
                return Nodes.eeg_right_top
            else:
                node = node.replace("EEG ", "")
                node = node.replace("eeg ", "")
        if "ref" in node.lower():
            node = node.replace("-Ref", "")
            node = node.replace("-REF", "")
        # if "ReOorAcht".lower() in node.lower() or "ReOorTop".lower() in node.lower():
        #     return Nodes.BTEright
        # if "LiOorAcht".lower() in node.lower() or "LiOorTop".lower() in node.lower():
        #     return Nodes.BTEleft
        if node.lower() == "ECG".lower():
            return Nodes.ecg
        if "unspec" in node.lower():
            node = node.replace("unspec", "")
            node = node.replace("Unspec", "")
        return node

    @staticmethod
    def match_nodes(nodes: list[str], possibilities: list[str]) -> list[str]:
        result = []
        # Ignore case and whitespace in the nodes
        stripped_nodes = [node.strip().lower() for node in nodes]
        stripped_possibilities = [possibility.strip().lower() for possibility in possibilities]

        for i, stripped_node in enumerate(stripped_nodes):
            if stripped_node in stripped_possibilities:
                index = stripped_possibilities.index(stripped_node)
                result.append(possibilities[index])
                continue
            formatted_node = Nodes.format_node_name(stripped_node).lower().strip()
            if formatted_node in stripped_possibilities:
                index = stripped_possibilities.index(formatted_node)
                result.append(possibilities[index])
            else:
                # If the node is not found, just leave it as is
                result.append(nodes[i])

        assert len(set(result)) == len(result), "Duplicate nodes found in the list"
        return result


class Locations:
    coimbra = "Coimbra_University_Hospital"
    freiburg = "Freiburg_University_Medical_Center"
    karolinska = "Karolinska_Institute"
    leuven_adult = "University_Hospital_Leuven_Adult"
    leuven_pediatric = "University_Hospital_Leuven_Pediatric"
    aachen = "University_of_Aachen"

    @staticmethod
    def to_acronym(loc):
        acronyms = {
            Locations.coimbra: "COI",
            Locations.freiburg: "FRB",
            Locations.karolinska: "KAR",
            Locations.leuven_adult: "LEU-AD",
            Locations.leuven_pediatric: "LEU-PE",
            Locations.aachen: "AAC"
        }
        return acronyms.get(loc, loc)

class Keys:
    minirocketLR = "MiniRocketLR"

class Paths:
    remote_data_path = "/cw/dtaidata/ml/2025-Epilepsy"
    local_data_path = "/media/loren/Seagate Basic/Epilepsy use case"  # path to dataset
    remote_save_dir = "/cw/dtailocal/loren/2025-Epilepsy/net/save_dir"
    local_save_dir = "net/save_dir"  # save directory of intermediate and output files


evaluation_metrics = {"roc_auc": auroc_score,
                      "score": get_sens_FA_score}
