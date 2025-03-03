import os
from typing import List, Tuple
import pandas as pd
import pyedflib


class Annotation:
    """ Class to store seizure annotations as read in the tsv annotation files from the SeizeIT2 BIDS dataset.
    """
    def __init__(
        self,
        events: List[Tuple[int, int]],
        type: List[str],
        lateralization: List[str],
        localization: List[str],
        vigilance: List[str],
        rec_duration: float
    ):
        """Initiate an annotation instance

        Args:
            events (List([int, int])): list of tuples where each element contains the start and stop times in seconds of the event
            type (List[str]): list of event types according to the dataset's events dictionary (events.json).
            lateralization (List[str]): list of lateralization characteristics of the events according to the dataset's events dictionary (events.json).
            localization (List[str]): list of localization characteristics of the events according to the dataset's events dictionary (events.json).
            vigilance (List[str]): list of vigilance characteristics of the events according to the dataset's events dictionary (events.json).

        Returns:
            Annotation: returns an Annotation instance containing the events of the recording.
        """
        self.events = events
        self.types = type
        self.lateralization = lateralization
        self.localization = localization
        self.vigilance = vigilance
        self.rec_duration = rec_duration

    @classmethod
    def loadAnnotation(
        cls,
        annotation_path: str,
        recording: List[str],
    ):
        szEvents = list()
        szTypes = list()
        szLat = list()
        szLoc = list()
        szVig = list()

        path = os.path.join(annotation_path, recording[0], recording[1])
        tsv_suffices = [suffix.split('_')[-1] for suffix in os.listdir(path) if suffix.endswith('.tsv') and suffix.startswith(f'{recording[1]}_{recording[2]}_')]
        if len(tsv_suffices) > 1:
            raise ValueError(f'Multiple tsv files found for recording {recording[0]} {recording[1]} {recording[2]}')
        tsvFile = os.path.join(path, f'{recording[1]}_{recording[2]}_{tsv_suffices[0]}')
        df = pd.read_csv(tsvFile, delimiter="\t", skiprows=4)
        for i, e in df.iterrows():
            if e['class'] == 'seizure' and e['main type'] == 'focal':
                szEvents.append([e['start'], e['stop']])
                szTypes.append(e['main type'])
                szLat.append(e['lateralization'])
                szLoc.append(e['localization'])
                szVig.append(e['vigilance'])
        # durs = e['stop'] - e['start']
        edf_path = os.path.join(path, f'{recording[1]}_{recording[2]}.edf')
        with pyedflib.EdfReader(edf_path) as edf:
            durs = edf.file_duration

        return cls(
            szEvents,
            szTypes,
            szLat,
            szLoc,
            szVig,
            durs,
        )
