import json


def bp2cl(bp):
    BASEPAIR_LENGTH = 0.00034  # um
    return bp * BASEPAIR_LENGTH


def write_config(fname, h_est, p_est):
    with open(fname, 'w') as f:
        json.dump({'handles': h_est, 'protein': p_est}, f, indent=4)


def read_config(fname):
    with open(fname) as f:
        return json.load(f)
