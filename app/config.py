# Shared configurations for PM2.5 Early Warning System

IS_HAZE_MONTHS = [1, 2, 3, 4]
PM25_THRESHOLDS = [25, 37.5, 50, 75]
FIRMS_BBOX_THAILAND = "97.3,17.5,102.5,20.5"
FIRMS_BBOX_GMS = "92.0,13.0,106.0,26.0"
FORECAST_DAYS = 7
NOTIFY_THRESHOLD_MODERATE = 37.5
NOTIFY_THRESHOLD_HIGH = 50
NOTIFY_THRESHOLD_DANGER = 75

PROVINCE_COORDS = {
    "Chiang Mai":   {"lat": 18.7883, "lon": 98.9853},
    "Chiang Rai":   {"lat": 19.9105, "lon": 99.8253},
    "Mae Hong Son": {"lat": 19.3003, "lon": 97.9654},
    "Lamphun":      {"lat": 18.5745, "lon": 99.0087},
    "Lampang":      {"lat": 18.2888, "lon": 99.4930},
    "Phayao":       {"lat": 19.1666, "lon": 99.9022},
    "Phrae":        {"lat": 18.1446, "lon": 100.1403},
    "Nan":          {"lat": 18.7756, "lon": 100.7730},
}

PROVINCES = sorted(list(PROVINCE_COORDS.keys()))

PROVINCE_LABELS = {p: i for i, p in enumerate(PROVINCES)}

PROVINCE_MEAN_MAP = {
    'Chiang Mai':   21.587774223034735,
    'Chiang Rai':   19.978260207190736,
    'Lampang':      18.251279707495428,
    'Lamphun':      17.79868982327849,
    'Mae Hong Son': 12.911837294332724,
    'Nan':          18.212416209628277,
    'Phayao':       17.143692870201097,
    'Phrae':        17.761144119439365,
}
