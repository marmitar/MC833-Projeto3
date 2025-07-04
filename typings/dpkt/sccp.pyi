

from . import dpkt

"""Cisco Skinny Client Control Protocol."""
KEYPAD_BUTTON = ...
OFF_HOOK = ...
ON_HOOK = ...
OPEN_RECEIVE_CHANNEL_ACK = ...
START_TONE = ...
STOP_TONE = ...
SET_LAMP = ...
SET_SPEAKER_MODE = ...
START_MEDIA_TRANSMIT = ...
STOP_MEDIA_TRANSMIT = ...
CALL_INFO = ...
DEFINE_TIME_DATE = ...
DISPLAY_TEXT = ...
OPEN_RECEIVE_CHANNEL = ...
CLOSE_RECEIVE_CHANNEL = ...
SELECT_SOFTKEYS = ...
CALL_STATE = ...
DISPLAY_PROMPT_STATUS = ...
CLEAR_PROMPT_STATUS = ...
ACTIVATE_CALL_PLANE = ...
class ActivateCallPlane(dpkt.Packet):
    __byte_order__ = ...
    __hdr__ = ...


class CallInfo(dpkt.Packet):
    __byte_order__ = ...
    __hdr__ = ...


class CallState(dpkt.Packet):
    __byte_order__ = ...
    __hdr__ = ...


class ClearPromptStatus(dpkt.Packet):
    __byte_order__ = ...
    __hdr__ = ...


class CloseReceiveChannel(dpkt.Packet):
    __byte_order__ = ...
    __hdr__ = ...


class DisplayPromptStatus(dpkt.Packet):
    __byte_order__ = ...
    __hdr__ = ...


class DisplayText(dpkt.Packet):
    __byte_order__ = ...
    __hdr__ = ...


class KeypadButton(dpkt.Packet):
    __byte_order__ = ...
    __hdr__ = ...


class OpenReceiveChannel(dpkt.Packet):
    __byte_order__ = ...
    __hdr__ = ...


class OpenReceiveChannelAck(dpkt.Packet):
    __byte_order__ = ...
    __hdr__ = ...


class SelectStartKeys(dpkt.Packet):
    __byte_order__ = ...
    __hdr__ = ...


class SetLamp(dpkt.Packet):
    __byte_order__ = ...
    __hdr__ = ...


class SetSpeakerMode(dpkt.Packet):
    __byte_order__ = ...
    __hdr__ = ...


class StartMediaTransmission(dpkt.Packet):
    __byte_order__ = ...
    __hdr__ = ...


class StartTone(dpkt.Packet):
    __byte_order__ = ...
    __hdr__ = ...


class StopMediaTransmission(dpkt.Packet):
    __byte_order__ = ...
    __hdr__ = ...


class SCCP(dpkt.Packet):

    __byte_order__ = ...
    __hdr__ = ...
    _msgsw = ...
    def unpack(self, buf): # -> None:
        ...



def test_sccp(): # -> None:
    ...
