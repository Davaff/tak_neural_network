# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tak/proto/taktician.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x19tak/proto/taktician.proto\x12\ttak.proto\"B\n\x0e\x41nalyzeRequest\x12\x10\n\x08position\x18\x01 \x01(\t\x12\r\n\x05\x64\x65pth\x18\x02 \x01(\x05\x12\x0f\n\x07precise\x18\x03 \x01(\x08\";\n\x0f\x41nalyzeResponse\x12\n\n\x02pv\x18\x01 \x03(\t\x12\r\n\x05value\x18\x02 \x01(\x03\x12\r\n\x05\x64\x65pth\x18\x03 \x01(\x05\"2\n\x13\x43\x61nonicalizeRequest\x12\x0c\n\x04size\x18\x01 \x01(\x05\x12\r\n\x05moves\x18\x02 \x03(\t\"%\n\x14\x43\x61nonicalizeResponse\x12\r\n\x05moves\x18\x01 \x03(\t\"*\n\x16IsPositionInTakRequest\x12\x10\n\x08position\x18\x01 \x01(\t\"9\n\x17IsPositionInTakResponse\x12\r\n\x05inTak\x18\x01 \x01(\x08\x12\x0f\n\x07takMove\x18\x02 \x01(\t2\xfe\x01\n\tTaktician\x12\x42\n\x07\x41nalyze\x12\x19.tak.proto.AnalyzeRequest\x1a\x1a.tak.proto.AnalyzeResponse\"\x00\x12Q\n\x0c\x43\x61nonicalize\x12\x1e.tak.proto.CanonicalizeRequest\x1a\x1f.tak.proto.CanonicalizeResponse\"\x00\x12Z\n\x0fIsPositionInTak\x12!.tak.proto.IsPositionInTakRequest\x1a\".tak.proto.IsPositionInTakResponse\"\x00\x42!Z\x1fgithub.com/nelhage/taktician/pbb\x06proto3')



_ANALYZEREQUEST = DESCRIPTOR.message_types_by_name['AnalyzeRequest']
_ANALYZERESPONSE = DESCRIPTOR.message_types_by_name['AnalyzeResponse']
_CANONICALIZEREQUEST = DESCRIPTOR.message_types_by_name['CanonicalizeRequest']
_CANONICALIZERESPONSE = DESCRIPTOR.message_types_by_name['CanonicalizeResponse']
_ISPOSITIONINTAKREQUEST = DESCRIPTOR.message_types_by_name['IsPositionInTakRequest']
_ISPOSITIONINTAKRESPONSE = DESCRIPTOR.message_types_by_name['IsPositionInTakResponse']
AnalyzeRequest = _reflection.GeneratedProtocolMessageType('AnalyzeRequest', (_message.Message,), {
  'DESCRIPTOR' : _ANALYZEREQUEST,
  '__module__' : 'tak.proto.taktician_pb2'
  # @@protoc_insertion_point(class_scope:tak.proto.AnalyzeRequest)
  })
_sym_db.RegisterMessage(AnalyzeRequest)

AnalyzeResponse = _reflection.GeneratedProtocolMessageType('AnalyzeResponse', (_message.Message,), {
  'DESCRIPTOR' : _ANALYZERESPONSE,
  '__module__' : 'tak.proto.taktician_pb2'
  # @@protoc_insertion_point(class_scope:tak.proto.AnalyzeResponse)
  })
_sym_db.RegisterMessage(AnalyzeResponse)

CanonicalizeRequest = _reflection.GeneratedProtocolMessageType('CanonicalizeRequest', (_message.Message,), {
  'DESCRIPTOR' : _CANONICALIZEREQUEST,
  '__module__' : 'tak.proto.taktician_pb2'
  # @@protoc_insertion_point(class_scope:tak.proto.CanonicalizeRequest)
  })
_sym_db.RegisterMessage(CanonicalizeRequest)

CanonicalizeResponse = _reflection.GeneratedProtocolMessageType('CanonicalizeResponse', (_message.Message,), {
  'DESCRIPTOR' : _CANONICALIZERESPONSE,
  '__module__' : 'tak.proto.taktician_pb2'
  # @@protoc_insertion_point(class_scope:tak.proto.CanonicalizeResponse)
  })
_sym_db.RegisterMessage(CanonicalizeResponse)

IsPositionInTakRequest = _reflection.GeneratedProtocolMessageType('IsPositionInTakRequest', (_message.Message,), {
  'DESCRIPTOR' : _ISPOSITIONINTAKREQUEST,
  '__module__' : 'tak.proto.taktician_pb2'
  # @@protoc_insertion_point(class_scope:tak.proto.IsPositionInTakRequest)
  })
_sym_db.RegisterMessage(IsPositionInTakRequest)

IsPositionInTakResponse = _reflection.GeneratedProtocolMessageType('IsPositionInTakResponse', (_message.Message,), {
  'DESCRIPTOR' : _ISPOSITIONINTAKRESPONSE,
  '__module__' : 'tak.proto.taktician_pb2'
  # @@protoc_insertion_point(class_scope:tak.proto.IsPositionInTakResponse)
  })
_sym_db.RegisterMessage(IsPositionInTakResponse)

_TAKTICIAN = DESCRIPTOR.services_by_name['Taktician']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z\037github.com/nelhage/taktician/pb'
  _ANALYZEREQUEST._serialized_start=40
  _ANALYZEREQUEST._serialized_end=106
  _ANALYZERESPONSE._serialized_start=108
  _ANALYZERESPONSE._serialized_end=167
  _CANONICALIZEREQUEST._serialized_start=169
  _CANONICALIZEREQUEST._serialized_end=219
  _CANONICALIZERESPONSE._serialized_start=221
  _CANONICALIZERESPONSE._serialized_end=258
  _ISPOSITIONINTAKREQUEST._serialized_start=260
  _ISPOSITIONINTAKREQUEST._serialized_end=302
  _ISPOSITIONINTAKRESPONSE._serialized_start=304
  _ISPOSITIONINTAKRESPONSE._serialized_end=361
  _TAKTICIAN._serialized_start=364
  _TAKTICIAN._serialized_end=618
# @@protoc_insertion_point(module_scope)
