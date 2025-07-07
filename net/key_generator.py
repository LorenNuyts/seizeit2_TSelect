from typing import List

import numpy as np
import random
from tqdm import tqdm
from data.annotation import Annotation


def generate_data_keys_sequential(config, recs_list, verbose=True):
    """Create data segment keys in a sequential time manner. The keys are 4 element lists corresponding to the file index in the 'recs_list', the start and stop in seconds of the segment and it's label.

        Args:
            config (cls): config object with the experiment's parameters.
            recs_list (list[list[str]]): a list of recording IDs in the format [sub-xxx, run-xx]
        Returns:
            segments: a list of data segment keys with [recording index, start, stop, label]
    """
    
    segments = []

    for idx, f in tqdm(enumerate(recs_list), disable = not verbose):
        annotations = Annotation.loadAnnotation(config.data_path, f)

        if not annotations.events:
            n_segs = int(np.floor((np.floor(annotations.rec_duration) - config.frame)/config.stride))
            seg_start = np.arange(0, n_segs)*config.stride
            seg_stop = seg_start + config.frame

            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
        else:
            if len(annotations.events) == 1:
                ev = annotations.events[0]
                n_segs = int(np.floor((ev[0])/config.stride)-1)
                seg_start = np.arange(0, n_segs)*config.stride
                seg_stop = seg_start + config.frame
                segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                n_segs = int(np.floor((ev[1] - ev[0])/config.stride) + 1)
                seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                seg_stop = seg_start + config.frame
                segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

                n_segs = int(np.floor(np.floor(annotations.rec_duration - ev[1])/config.stride)-1)
                seg_start = np.arange(0, n_segs)*config.stride + ev[1]
                seg_stop = seg_start + config.frame
                segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
            else:
                for e, ev in enumerate(annotations.events):
                    # If it is the first event
                    if e == 0:
                        n_segs = int(np.floor((ev[0])/config.stride)-1)
                        if n_segs < 0:
                            n_segs = 0
                        seg_start = np.arange(0, n_segs)*config.stride
                        seg_stop = seg_start + config.frame
                        segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                        n_segs = int(np.floor((ev[1] - ev[0])/config.stride)+1)
                        seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                        if np.sum(seg_start<0) > 0:
                            n_segs -= np.sum(seg_start<0)
                            seg_start = seg_start[seg_start>=0]
                        seg_stop = seg_start + config.frame
                        segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

                    # If it is not the last event and not the first event
                    elif e != len(annotations.events)-1:
                        prev_event = annotations.events[e-1]
                        if ev[0] < segments[-1][2]:
                            # If the start of the event is before the end of the previous segment, start from the end
                            # of the previous segment
                            start_event_corrected = segments[-1][2]
                            n_segs = int(np.floor((ev[1] - start_event_corrected) / config.stride) + 1)
                            seg_start = np.arange(0, n_segs) * config.stride + start_event_corrected - config.stride
                            seg_stop = seg_start + config.frame
                            segments.extend(np.column_stack(([idx] * n_segs, seg_start, seg_stop, np.ones(n_segs))))
                        else:
                            n_segs = int(np.floor((ev[0] - prev_event[1])/config.stride)-1)
                            seg_start = np.arange(0, n_segs)*config.stride + prev_event[1]
                            seg_stop = seg_start + config.frame
                            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                            n_segs = int(np.floor((ev[1] - ev[0])/config.stride)+1)
                            seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                            seg_stop = seg_start + config.frame
                            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

                    # If it is the last event
                    elif e == len(annotations.events)-1:
                        prev_event = annotations.events[e-1]
                        if ev[0] < segments[-1][2]:
                            # If the start of the event is before the end of the previous segment, start from the end
                            # of the previous segment
                            start_event_corrected = segments[-1][2]
                            n_segs = int(np.floor((ev[1] - start_event_corrected) / config.stride) + 1)
                            seg_start = np.arange(0, n_segs) * config.stride + start_event_corrected - config.stride
                            seg_stop = seg_start + config.frame
                            segments.extend(np.column_stack(([idx] * n_segs, seg_start, seg_stop, np.ones(n_segs))))
                        else:
                            n_segs = int(np.floor((ev[0] - prev_event[1])/config.stride)-1)
                            seg_start = np.arange(0, n_segs)*config.stride + prev_event[1]
                            seg_stop = seg_start + config.frame
                            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                            n_segs = int(np.floor((ev[1] - ev[0])/config.stride)+1)
                            seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                            seg_stop = seg_start + config.frame
                            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

                        n_segs = int(np.floor((annotations.rec_duration - ev[1])/config.stride)-1)
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride + ev[1]
                            seg_stop = seg_start + config.frame
                            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

    return segments


def generate_data_keys_sequential_window(config, recs_list: List[List[str]], t_add: int):
    """Create data segment keys in a sequential time manner with a window of 2*t_add (where t_add is in seconds). Specific key generator for the validation data of the current framework.

        Args:
            config (cls): config object with the experiment's parameters.
            recs_list (list[list[str]]): a list of recording IDs in the format [sub-xxx, run-xx]
            t_add: time to add before and after the center time point of the event.
        Returns:
            segments: a list of data segment keys with [recording index, start, stop, label]
    """
    # Helper function to adapt the time interval around the event that is included in the segments.
    def adapt_bounds(start, end, to_add_start, to_add_end, to_add_default, frame, rec_duration):
        # Check if the event exceeds the recording duration: end
        if end + to_add_default > np.floor(rec_duration) - frame:
            to_add_end = np.floor(rec_duration) - end - frame  # Compute the surplus at the end
            to_add_start = to_add_default + to_add_default - to_add_end  # Add the surplus at the end to the start
        # Check if the event exceeds the recording duration: start
        if start - to_add_default < 0:
            to_add_start = start - 1  # Compute the surplus at the start
            to_add_end = to_add_default + to_add_default - to_add_start  # Add the surplus at the start to the end
        if to_add_end + to_add_start + end - start > t_add * 2:
            to_add_end = to_add_end - (to_add_end + to_add_start + end - start - t_add * 2)
        elif to_add_end + to_add_start + end - start < t_add * 2:
            # If the event exceeds the end or the start of the recording
            if to_add_end == np.floor(rec_duration) - end - frame:
                to_add_start += (
                        t_add * 2 - (to_add_end + to_add_start + end - start))  # Add what is missing to the start
            # If the event exceeds the start of the recording
            elif to_add_start == start - 1:
                to_add_end += (
                        t_add * 2 - (to_add_end + to_add_start + end - start))  # Add what is missing to the end
            else:
                to_add_end += (
                        t_add * 2 - (to_add_end + to_add_start + end - start))  # Just add what is missing to the end
        # Check that the total interval has a length of 2*t_add
        if to_add_end + to_add_start + end - start != t_add * 2:
            print('bad segmentation!!!')
        return to_add_start, to_add_end

    # Start main function
    segments = []

    for idx, f in tqdm(enumerate(recs_list)):
        print("Processing file:", f)
        annotations = Annotation.loadAnnotation(config.data_path, f)

        if annotations.rec_duration < 2*t_add:
            print('short file: ' + f[0] + ' ' + f[1])

        if annotations.events:
            if len(annotations.events) == 1:
                ev: (int, int) = annotations.events[0]


                # t_add_ev is the amount of background signal in an interval of t_add
                if t_add*2 < ev[1]-ev[0]:
                    print('check batches!!!')
                    to_add_ev = 30
                else:
                    to_add_ev = t_add - round((ev[1]-ev[0])/2)

                to_add_plus = to_add_ev
                to_add_minus = to_add_ev

                to_add_minus, to_add_plus = adapt_bounds(ev[0], ev[1], to_add_minus, to_add_plus, to_add_ev,
                                                         config.frame, annotations.rec_duration)

                new_segments = segment_sequential_window(ev[0], ev[1], to_add_minus, to_add_plus, config.stride,
                                                            config.frame, idx)
                if len(new_segments) != 2*t_add:
                    print('wrong nr segs')
                segments.extend(new_segments)
            else: # If there are multiple events in the recording
                end_rec = False
                end_seg = 0
                for i, ev in enumerate(annotations.events):
                    skip = False
                    if t_add*2 < ev[1]-ev[0]:
                        print('check batches!!!')
                        to_add_ev = 30
                    else:
                        to_add_ev = t_add - round((ev[1]-ev[0])/2)

                    if i == 0:   # The first event
                        to_add_plus = to_add_ev
                        to_add_minus = to_add_ev

                        if ev[0] - to_add_ev < 0:
                            to_add_minus = ev[0]-1
                            to_add_plus = to_add_ev + (to_add_ev - ev[0]) + 1

                        end_seg = to_add_plus

                    else:
                        # If the start of the event is after the end of the previous event
                        if ev[0] > end_seg:
                            # If the two events are sufficiently far apart, the params are reset
                            if ev[0] - to_add_ev > end_seg:
                                to_add_minus = to_add_ev
                                to_add_plus = to_add_ev
                            else:
                                to_add_minus =  ev[0] - end_seg
                                to_add_plus = 2*to_add_ev - to_add_minus
                        else:
                            # Part of the current events overlaps with the previous event, but not entirely
                            if ev[1] > end_seg:
                                print('check boundary case')
                            # The entire current event overlaps with the previous event, so skip it
                            else:
                                skip = True

                        end_seg = ev[1] + to_add_plus
                    
                    if end_seg > np.floor(annotations.rec_duration)-config.frame - t_add*2:
                        end_rec = True
                    

                    if not skip and not end_rec:
                        to_add_minus, to_add_plus = adapt_bounds(ev[0], ev[1], to_add_minus, to_add_plus, to_add_ev,
                                                                 config.frame, annotations.rec_duration)

                        new_segments = segment_sequential_window(ev[0], ev[1], to_add_minus, to_add_plus, config.stride,
                                                                 config.frame, idx)
                        if len(new_segments) != 2 * t_add:
                            print('wrong nr segs')
                        segments.extend(new_segments)

                    elif skip and not end_rec:

                        n_segs = int(np.floor((ev[1] - ev[0])/config.stride) + 1)
                        seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                        # Find the segments that overlap with the event
                        idxs_seiz = [i for i,x in enumerate(segments) if x[1] in seg_start]
                        for ii in idxs_seiz:
                            segments[ii][3] = 1    # Make sure that the label is 1 for the segments that overlap with the event


    return segments


def segment_sequential_window(start, stop, to_add_start, to_add_end, stride, frame, rec_index):
    segments = []
    segs_nr = 0
    n_segs = int(np.floor((start - (start - to_add_start)) / stride) - 1)
    if n_segs < 0:
        n_segs = 0
    seg_start = np.arange(0, n_segs) * stride + start - to_add_start
    seg_stop = seg_start + frame
    segments.extend(np.column_stack(([rec_index] * n_segs, seg_start, seg_stop, np.zeros(n_segs))))
    segs_nr += n_segs
    n_segs = int(np.floor((stop - start) / stride) + 1)
    seg_start = np.arange(0, n_segs) * stride + start - stride
    seg_stop = seg_start + frame
    segments.extend(np.column_stack(([rec_index] * n_segs, seg_start, seg_stop, np.ones(n_segs))))
    segs_nr += n_segs
    n_segs = int(np.floor(np.floor(stop + to_add_end - stop) / stride))
    if n_segs < 0:
        n_segs = 0
    seg_start = np.arange(0, n_segs) * stride + stop
    seg_stop = seg_start + frame
    segments.extend(np.column_stack(([rec_index] * n_segs, seg_start, seg_stop, np.zeros(n_segs))))
    segs_nr += n_segs
    return segments


def generate_data_keys_subsample(config, recs_list):
    """Create data segment keys by subsampling the data, including all seizure segments (Ns) and config.factor*Ns non-seizure segments.

        Args:
            config (cls): config object with the experiment's parameters.
            recs_list (list[list[str]]): a list of recording IDs in the format [SUBJ-x-xxx, rxx]
        Returns:
            segments: a list of data segment keys with [recording index, start, stop, label]
    """
    
    segments_S = []
    segments_NS = []

    for idx, f in tqdm(enumerate(recs_list)):
        annotations = Annotation.loadAnnotation(config.data_path, f)

        if not annotations.events:
            n_segs = int(np.floor((np.floor(annotations.rec_duration) - config.frame)/config.stride))
            seg_start = np.arange(0, n_segs)*config.stride
            seg_stop = seg_start + config.frame

            segments_NS.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
        else:
            for e, ev in enumerate(annotations.events):
                n_segs = int(((ev[1]+config.frame*(1-config.boundary))-(ev[0]-config.frame*(1-config.boundary))-config.frame)/config.stride_s)
                seg_start = np.arange(0, n_segs)*config.stride_s + ev[0]-config.frame*(1-config.boundary)
                seg_stop = seg_start + config.frame
                segments_S.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

                if e == 0:
                    n_segs = int(np.floor((ev[0])/config.stride)-1)
                    seg_start = np.arange(0, n_segs)*config.stride
                    seg_stop = seg_start + config.frame
                    if n_segs < 0:
                        n_segs = 0
                    segments_NS.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                else:
                    n_segs = int(np.floor((ev[0] - annotations.events[e-1][1])/config.stride)-1)
                    if n_segs < 0:
                        n_segs = 0
                    seg_start = np.arange(0, n_segs)*config.stride + annotations.events[e-1][1]
                    seg_stop = seg_start + config.frame
                    segments_NS.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                if e == len(annotations.events)-1:
                    n_segs = int(np.floor((np.floor(annotations.rec_duration) - ev[1])/config.stride)-1)
                    seg_start = np.arange(0, n_segs)*config.stride + ev[1]
                    seg_stop = seg_start + config.frame
                    segments_NS.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
        
    segments_S.extend(random.sample(segments_NS, config.factor*len(segments_S)))
    random.shuffle(segments_S)

    return segments_S
