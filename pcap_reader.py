import struct
import os
from typing import Generator, Tuple, Optional, Callable

class PcapReader:
    """
    High-performance pure Python PCAP file reader for UDP LiDAR data.
    Supports standard microsecond/nanosecond PCAP formats, big/little endian,
    VLAN tags, IPv4, and UDP extraction.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"PCAP file not found: {filepath}")
        self.filesize = os.path.getsize(filepath)

    def read_packets(self,
                     filter_ports=(2368, 8603),
                     progress_callback: Optional[Callable[[float], None]] = None
                     ) -> Generator[Tuple[float, str, int, bytes], None, None]:
        """
        Yields (timestamp_sec, src_ip, dst_port, payload_bytes).
        Periodically calls progress_callback(fraction: 0.0 ~ 1.0) if provided.
        """
        with open(self.filepath, 'rb') as f:
            magic_bytes = f.read(4)
            if len(magic_bytes) < 4:
                return

            if magic_bytes == b'\xd4\xc3\xb2\xa1': # LE microsecond
                endian = '<'
                is_nanosec = False
            elif magic_bytes == b'\xa1\xb2\xc3\xd4': # BE microsecond
                endian = '>'
                is_nanosec = False
            elif magic_bytes == b'\x4d\x3c\xb2\xa1': # LE nanosecond
                endian = '<'
                is_nanosec = True
            elif magic_bytes == b'\xa1\xb2\x3c\x4d': # BE nanosecond
                endian = '>'
                is_nanosec = True
            else:
                endian = '<'
                is_nanosec = False

            f.seek(0)
            global_hdr = f.read(24)
            if len(global_hdr) < 24:
                return

            linktype = struct.unpack(endian + 'I', global_hdr[20:24])[0]
            ts_scale = 1e-9 if is_nanosec else 1e-6
            pkt_hdr_struct = struct.Struct(endian + 'IIII')
            pkt_count = 0

            while True:
                hdr_bytes = f.read(16)
                if len(hdr_bytes) < 16:
                    break
                ts_sec, ts_frac, incl_len, orig_len = pkt_hdr_struct.unpack(hdr_bytes)
                pkt_data = f.read(incl_len)
                if len(pkt_data) < incl_len:
                    break

                pkt_count += 1
                if progress_callback and (pkt_count % 500 == 0) and self.filesize > 0:
                    progress_callback(min(1.0, f.tell() / self.filesize))

                timestamp = ts_sec + ts_frac * ts_scale

                # Parse link layer
                offset = 0
                if linktype == 1: # Ethernet
                    if incl_len < 14:
                        continue
                    proto = struct.unpack('!H', pkt_data[12:14])[0]
                    offset = 14
                    while proto in (0x8100, 0x88a8, 0x9100):
                        if incl_len < offset + 4:
                            break
                        proto = struct.unpack('!H', pkt_data[offset+2:offset+4])[0]
                        offset += 4
                elif linktype == 113: # Linux cooked v1
                    if incl_len < 16:
                        continue
                    proto = struct.unpack('!H', pkt_data[14:16])[0]
                    offset = 16
                elif linktype == 276: # Linux cooked v2
                    if incl_len < 20:
                        continue
                    proto = struct.unpack('!H', pkt_data[0:2])[0]
                    offset = 20
                elif linktype in (101, 228): # Raw IPv4
                    proto = 0x0800
                    offset = 0
                else:
                    continue

                if proto != 0x0800:
                    continue

                if incl_len < offset + 20:
                    continue
                ver_ihl = pkt_data[offset]
                if (ver_ihl >> 4) != 4:
                    continue
                ihl = (ver_ihl & 0x0F) * 4
                ip_proto = pkt_data[offset + 9]
                if ip_proto != 17:
                    continue

                frag_offset = struct.unpack('!H', pkt_data[offset+6:offset+8])[0] & 0x1FFF
                if frag_offset != 0:
                    continue

                udp_offset = offset + ihl
                if incl_len < udp_offset + 8:
                    continue

                src_port, dst_port, udp_len, _ = struct.unpack('!HHHH', pkt_data[udp_offset:udp_offset+8])
                if filter_ports and (dst_port not in filter_ports):
                    continue

                if udp_len < 8 or incl_len < udp_offset + udp_len:
                    continue

                payload = pkt_data[udp_offset+8 : udp_offset+udp_len]
                src_ip = f"{pkt_data[offset+12]}.{pkt_data[offset+13]}.{pkt_data[offset+14]}.{pkt_data[offset+15]}"

                yield (timestamp, src_ip, dst_port, payload)

            if progress_callback:
                progress_callback(1.0)
