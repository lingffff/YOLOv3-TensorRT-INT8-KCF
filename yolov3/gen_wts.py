import struct
import argparse
from models import *
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--tiny', action='store_true', help='use YOLOv3-tiny model')
opt = parser.parse_args()
if opt.tiny:
    cfg_path = 'redball/redball-tiny.cfg'
    output_path = './redball-tiny.wts'
else:
    cfg_path = 'redball/redball.cfg'
    output_path = './redball.wts'
model = Darknet(cfg_path)
weights = 'weights/best.pt'
device = torch_utils.select_device('0')
if weights.endswith('.pt'):  # pytorch format
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
else:  # darknet format
    load_darknet_weights(model, weights)
model = model.eval()

f = open(output_path, 'w')
f.write('{}\n'.format(len(model.state_dict().keys())))
print('Waiting...')
for k, v in model.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')
print('Done.')
