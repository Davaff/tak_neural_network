# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tak/proto/analysis.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x18tak/proto/analysis.proto\x12\ttak.proto\"#\n\x0f\x45valuateRequest\x12\x10\n\x08position\x18\x01 \x03(\x05\"O\n\x10\x45valuateResponse\x12\x12\n\nmove_probs\x18\x01 \x03(\x02\x12\r\n\x05value\x18\x02 \x01(\x02\x12\x18\n\x10move_probs_bytes\x18\x03 \x01(\x0c\x32Q\n\x08\x41nalysis\x12\x45\n\x08\x45valuate\x12\x1a.tak.proto.EvaluateRequest\x1a\x1b.tak.proto.EvaluateResponse\"\x00\x42\"Z github.com/nelhage/taktician/pb/b\x06proto3')



_EVALUATEREQUEST = DESCRIPTOR.message_types_by_name['EvaluateRequest']
_EVALUATERESPONSE = DESCRIPTOR.message_types_by_name['EvaluateResponse']
EvaluateRequest = _reflection.GeneratedProtocolMessageType('EvaluateRequest', (_message.Message,), {
  'DESCRIPTOR' : _EVALUATEREQUEST,
  '__module__' : 'tak.proto.analysis_pb2'
  # @@protoc_insertion_point(class_scope:tak.proto.EvaluateRequest)
  })
_sym_db.RegisterMessage(EvaluateRequest)

EvaluateResponse = _reflection.GeneratedProtocolMessageType('EvaluateResponse', (_message.Message,), {
  'DESCRIPTOR' : _EVALUATERESPONSE,
  '__module__' : 'tak.proto.analysis_pb2'
  # @@protoc_insertion_point(class_scope:tak.proto.EvaluateResponse)
  })
_sym_db.RegisterMessage(EvaluateResponse)

_ANALYSIS = DESCRIPTOR.services_by_name['Analysis']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z github.com/nelhage/taktician/pb/'
  _EVALUATEREQUEST._serialized_start=39
  _EVALUATEREQUEST._serialized_end=74
  _EVALUATERESPONSE._serialized_start=76
  _EVALUATERESPONSE._serialized_end=155
  _ANALYSIS._serialized_start=157
  _ANALYSIS._serialized_end=238
# @@protoc_insertion_point(module_scope)
