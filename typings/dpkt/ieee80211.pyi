

from . import dpkt

"""IEEE 802.11."""
MGMT_TYPE = ...
CTL_TYPE = ...
DATA_TYPE = ...
M_ASSOC_REQ = ...
M_ASSOC_RESP = ...
M_REASSOC_REQ = ...
M_REASSOC_RESP = ...
M_PROBE_REQ = ...
M_PROBE_RESP = ...
M_BEACON = ...
M_ATIM = ...
M_DISASSOC = ...
M_AUTH = ...
M_DEAUTH = ...
M_ACTION = ...
C_BLOCK_ACK_REQ = ...
C_BLOCK_ACK = ...
C_PS_POLL = ...
C_RTS = ...
C_CTS = ...
C_ACK = ...
C_CF_END = ...
C_CF_END_ACK = ...
D_DATA = ...
D_DATA_CF_ACK = ...
D_DATA_CF_POLL = ...
D_DATA_CF_ACK_POLL = ...
D_NULL = ...
D_CF_ACK = ...
D_CF_POLL = ...
D_CF_ACK_POLL = ...
D_QOS_DATA = ...
D_QOS_CF_ACK = ...
D_QOS_CF_POLL = ...
D_QOS_CF_ACK_POLL = ...
D_QOS_NULL = ...
D_QOS_CF_POLL_EMPTY = ...
TO_DS_FLAG = ...
FROM_DS_FLAG = ...
INTER_DS_FLAG = ...
_VERSION_MASK = ...
_TYPE_MASK = ...
_SUBTYPE_MASK = ...
_TO_DS_MASK = ...
_FROM_DS_MASK = ...
_MORE_FRAG_MASK = ...
_RETRY_MASK = ...
_PWR_MGT_MASK = ...
_MORE_DATA_MASK = ...
_WEP_MASK = ...
_ORDER_MASK = ...
_FRAGMENT_NUMBER_MASK = ...
_SEQUENCE_NUMBER_MASK = ...
_VERSION_SHIFT = ...
_TYPE_SHIFT = ...
_SUBTYPE_SHIFT = ...
_TO_DS_SHIFT = ...
_FROM_DS_SHIFT = ...
_MORE_FRAG_SHIFT = ...
_RETRY_SHIFT = ...
_PWR_MGT_SHIFT = ...
_MORE_DATA_SHIFT = ...
_WEP_SHIFT = ...
_ORDER_SHIFT = ...
_SEQUENCE_NUMBER_SHIFT = ...
IE_SSID = ...
IE_RATES = ...
IE_FH = ...
IE_DS = ...
IE_CF = ...
IE_TIM = ...
IE_IBSS = ...
IE_HT_CAPA = ...
IE_ESR = ...
IE_HT_INFO = ...
FCS_LENGTH = ...
FRAMES_WITH_CAPABILITY = ...
_ACK_POLICY_SHIFT = ...
_MULTI_TID_SHIFT = ...
_COMPRESSED_SHIFT = ...
_TID_SHIFT = ...
_ACK_POLICY_MASK = ...
_MULTI_TID_MASK = ...
_COMPRESSED_MASK = ...
_TID_MASK = ...
_COMPRESSED_BMP_LENGTH = ...
_BMP_LENGTH = ...
BLOCK_ACK = ...
BLOCK_ACK_CODE_REQUEST = ...
BLOCK_ACK_CODE_RESPONSE = ...
BLOCK_ACK_CODE_DELBA = ...
class IEEE80211(dpkt.Packet):

    __hdr__ = ...
    @property
    def version(self):
        ...

    @version.setter
    def version(self, val): # -> None:
        ...

    @property
    def type(self):
        ...

    @type.setter
    def type(self, val): # -> None:
        ...

    @property
    def subtype(self):
        ...

    @subtype.setter
    def subtype(self, val): # -> None:
        ...

    @property
    def to_ds(self):
        ...

    @to_ds.setter
    def to_ds(self, val): # -> None:
        ...

    @property
    def from_ds(self):
        ...

    @from_ds.setter
    def from_ds(self, val): # -> None:
        ...

    @property
    def more_frag(self):
        ...

    @more_frag.setter
    def more_frag(self, val): # -> None:
        ...

    @property
    def retry(self):
        ...

    @retry.setter
    def retry(self, val): # -> None:
        ...

    @property
    def pwr_mgt(self):
        ...

    @pwr_mgt.setter
    def pwr_mgt(self, val): # -> None:
        ...

    @property
    def more_data(self):
        ...

    @more_data.setter
    def more_data(self, val): # -> None:
        ...

    @property
    def wep(self):
        ...

    @wep.setter
    def wep(self, val): # -> None:
        ...

    @property
    def order(self):
        ...

    @order.setter
    def order(self, val): # -> None:
        ...

    def unpack_ies(self, buf): # -> None:
        ...

    class Capability:
        def __init__(self, field) -> None:
            ...



    def __init__(self, *args, **kwargs) -> None:
        ...

    def unpack(self, buf): # -> None:
        ...

    class BlockAckReq(dpkt.Packet):
        __hdr__ = ...


    class BlockAck(dpkt.Packet):
        __hdr__ = ...
        @property
        def compressed(self): # -> Any:
            ...

        @compressed.setter
        def compressed(self, val): # -> None:
            ...

        @property
        def ack_policy(self): # -> Any:
            ...

        @ack_policy.setter
        def ack_policy(self, val): # -> None:
            ...

        @property
        def multi_tid(self): # -> Any:
            ...

        @multi_tid.setter
        def multi_tid(self, val): # -> None:
            ...

        @property
        def tid(self): # -> Any:
            ...

        @tid.setter
        def tid(self, val): # -> None:
            ...

        def unpack(self, buf): # -> None:
            ...



    class _FragmentNumSeqNumMixin:
        @property
        def fragment_number(self): # -> Any:
            ...

        @property
        def sequence_number(self): # -> Any:
            ...



    class RTS(dpkt.Packet):
        __hdr__ = ...


    class CTS(dpkt.Packet):
        __hdr__ = ...


    class ACK(dpkt.Packet):
        __hdr__ = ...


    class CFEnd(dpkt.Packet):
        __hdr__ = ...


    class MGMT_Frame(dpkt.Packet, _FragmentNumSeqNumMixin):
        __hdr__ = ...


    class Beacon(dpkt.Packet):
        __hdr__ = ...
        def unpack(self, buf): # -> None:
            ...



    class Disassoc(dpkt.Packet):
        __hdr__ = ...


    class Assoc_Req(dpkt.Packet):
        __hdr__ = ...


    class Assoc_Resp(dpkt.Packet):
        __hdr__ = ...


    class Reassoc_Req(dpkt.Packet):
        __hdr__ = ...


    class Auth(dpkt.Packet):
        __hdr__ = ...


    class Deauth(dpkt.Packet):
        __hdr__ = ...


    class Action(dpkt.Packet):
        __hdr__ = ...
        def unpack(self, buf): # -> None:
            ...



    class BlockAckActionRequest(dpkt.Packet):
        __hdr__ = ...


    class BlockAckActionResponse(dpkt.Packet):
        __hdr__ = ...


    class BlockAckActionDelba(dpkt.Packet):
        __byte_order__ = ...
        __hdr__ = ...


    class Data(dpkt.Packet, _FragmentNumSeqNumMixin):
        __hdr__ = ...


    class DataFromDS(dpkt.Packet, _FragmentNumSeqNumMixin):
        __hdr__ = ...


    class DataToDS(dpkt.Packet, _FragmentNumSeqNumMixin):
        __hdr__ = ...


    class DataInterDS(dpkt.Packet, _FragmentNumSeqNumMixin):
        __hdr__ = ...


    class QoS_Data(dpkt.Packet):
        __hdr__ = ...


    class IE(dpkt.Packet):
        __hdr__ = ...
        def unpack(self, buf): # -> None:
            ...



    class FH(dpkt.Packet):
        __hdr__ = ...


    class DS(dpkt.Packet):
        __hdr__ = ...


    class CF(dpkt.Packet):
        __hdr__ = ...


    class TIM(dpkt.Packet):
        __hdr__ = ...
        def unpack(self, buf): # -> None:
            ...



    class IBSS(dpkt.Packet):
        __hdr__ = ...




def test_802211_ack(): # -> None:
    ...

def test_80211_beacon(): # -> None:
    ...

def test_80211_data(): # -> None:
    ...

def test_80211_data_qos(): # -> None:
    ...

def test_bug(): # -> None:
    ...

def test_data_ds(): # -> None:
    ...

def test_compressed_block_ack(): # -> None:
    ...

def test_action_block_ack_request(): # -> None:
    ...

def test_action_block_ack_response(): # -> None:
    ...

def test_action_block_ack_delete(): # -> None:
    ...

def test_ieee80211_properties(): # -> None:
    ...

def test_blockack_properties(): # -> None:
    ...

def test_ieee80211_unpack(): # -> None:
    ...

def test_blockack_unpack(): # -> None:
    ...

def test_action_unpack(): # -> None:
    ...

def test_beacon_unpack(): # -> None:
    ...

def test_fragment_and_sequence_values(): # -> None:
    ...
