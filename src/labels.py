from enum import IntEnum as _IE

class LabelsEnum(_IE):
    LANE_CHANGE = 0
    LANE_MERGE = 1
    OVERTAKE = 2
    BRAKING = 3
    TURN_INTENT = 4
    COLLISION = 5
    PEDESTRIAN_IN_ROAD = 6
    OBSTACLE_IN_ROAD = 7
    TRAFFIC_JAM = 8

def getMlbNames(encoded_mlb:int)->set[str]:
    labels = set()
    for label in LabelsEnum:
        if (encoded_mlb & (1 << label.value)) != 0:
            labels.add(label.name)
    return labels

def getLbName(label_idx:int)->str:
    try:
        return LabelsEnum(label_idx).name
    except ValueError:
        return "UNKNOWN_LABEL"
