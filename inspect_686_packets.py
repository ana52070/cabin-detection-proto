# -*- coding: utf-8 -*-
from tm16_decoder import TM16Decoder
import config

dec = TM16Decoder()
frames = dec.decode_pcap(config.DEFAULT_TRUCK_PCAP)

for i in [684, 685, 686, 687, 688]:
    f = frames[i]
    print(f"Py Frame {i}: pts={len(f.points_site)}, packets={f.packet_count}, t={f.timestamp:.4f}")
