# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: YOLOsvc.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='YOLOsvc.proto',
  package='YOLOsvc',
  syntax='proto3',
  serialized_options=b'\n\030io.grpc.examples.yolosvcB\014YOLOsvcProtoP\001\242\002\003RTG',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\rYOLOsvc.proto\x12\x07YOLOsvc\"\x11\n\x03RGB\x12\n\n\x02px\x18\x01 \x03(\x05\"%\n\x08PixelRow\x12\x19\n\x03row\x18\x01 \x03(\x0b\x32\x0c.YOLOsvc.RGB\"\'\n\x05Image\x12\x1e\n\x03img\x18\x01 \x03(\x0b\x32\x11.YOLOsvc.PixelRow\"i\n\nDetections\x12\x0c\n\x04xmin\x18\x01 \x03(\x02\x12\x0c\n\x04xmax\x18\x02 \x03(\x02\x12\x0c\n\x04ymin\x18\x03 \x03(\x02\x12\x0c\n\x04ymax\x18\x04 \x03(\x02\x12\x0f\n\x07\x63lasses\x18\x05 \x03(\x05\x12\x12\n\nconfidence\x18\x06 \x03(\x02\";\n\x08ImageB64\x12\x10\n\x08\x62\x36\x34image\x18\x01 \x01(\t\x12\r\n\x05width\x18\x02 \x01(\x05\x12\x0e\n\x06height\x18\x03 \x01(\x05\x32\x82\x01\n\x07YOLOsvc\x12\x38\n\x0fObjectDetection\x12\x0e.YOLOsvc.Image\x1a\x13.YOLOsvc.Detections\"\x00\x12=\n\x11ObjectDetectionV2\x12\x11.YOLOsvc.ImageB64\x1a\x13.YOLOsvc.Detections\"\x00\x42\x30\n\x18io.grpc.examples.yolosvcB\x0cYOLOsvcProtoP\x01\xa2\x02\x03RTGb\x06proto3'
)




_RGB = _descriptor.Descriptor(
  name='RGB',
  full_name='YOLOsvc.RGB',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='px', full_name='YOLOsvc.RGB.px', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=26,
  serialized_end=43,
)


_PIXELROW = _descriptor.Descriptor(
  name='PixelRow',
  full_name='YOLOsvc.PixelRow',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='row', full_name='YOLOsvc.PixelRow.row', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=45,
  serialized_end=82,
)


_IMAGE = _descriptor.Descriptor(
  name='Image',
  full_name='YOLOsvc.Image',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='img', full_name='YOLOsvc.Image.img', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=84,
  serialized_end=123,
)


_DETECTIONS = _descriptor.Descriptor(
  name='Detections',
  full_name='YOLOsvc.Detections',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='xmin', full_name='YOLOsvc.Detections.xmin', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='xmax', full_name='YOLOsvc.Detections.xmax', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ymin', full_name='YOLOsvc.Detections.ymin', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ymax', full_name='YOLOsvc.Detections.ymax', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='classes', full_name='YOLOsvc.Detections.classes', index=4,
      number=5, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='confidence', full_name='YOLOsvc.Detections.confidence', index=5,
      number=6, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=125,
  serialized_end=230,
)


_IMAGEB64 = _descriptor.Descriptor(
  name='ImageB64',
  full_name='YOLOsvc.ImageB64',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='b64image', full_name='YOLOsvc.ImageB64.b64image', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='width', full_name='YOLOsvc.ImageB64.width', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='height', full_name='YOLOsvc.ImageB64.height', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=232,
  serialized_end=291,
)

_PIXELROW.fields_by_name['row'].message_type = _RGB
_IMAGE.fields_by_name['img'].message_type = _PIXELROW
DESCRIPTOR.message_types_by_name['RGB'] = _RGB
DESCRIPTOR.message_types_by_name['PixelRow'] = _PIXELROW
DESCRIPTOR.message_types_by_name['Image'] = _IMAGE
DESCRIPTOR.message_types_by_name['Detections'] = _DETECTIONS
DESCRIPTOR.message_types_by_name['ImageB64'] = _IMAGEB64
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RGB = _reflection.GeneratedProtocolMessageType('RGB', (_message.Message,), {
  'DESCRIPTOR' : _RGB,
  '__module__' : 'YOLOsvc_pb2'
  # @@protoc_insertion_point(class_scope:YOLOsvc.RGB)
  })
_sym_db.RegisterMessage(RGB)

PixelRow = _reflection.GeneratedProtocolMessageType('PixelRow', (_message.Message,), {
  'DESCRIPTOR' : _PIXELROW,
  '__module__' : 'YOLOsvc_pb2'
  # @@protoc_insertion_point(class_scope:YOLOsvc.PixelRow)
  })
_sym_db.RegisterMessage(PixelRow)

Image = _reflection.GeneratedProtocolMessageType('Image', (_message.Message,), {
  'DESCRIPTOR' : _IMAGE,
  '__module__' : 'YOLOsvc_pb2'
  # @@protoc_insertion_point(class_scope:YOLOsvc.Image)
  })
_sym_db.RegisterMessage(Image)

Detections = _reflection.GeneratedProtocolMessageType('Detections', (_message.Message,), {
  'DESCRIPTOR' : _DETECTIONS,
  '__module__' : 'YOLOsvc_pb2'
  # @@protoc_insertion_point(class_scope:YOLOsvc.Detections)
  })
_sym_db.RegisterMessage(Detections)

ImageB64 = _reflection.GeneratedProtocolMessageType('ImageB64', (_message.Message,), {
  'DESCRIPTOR' : _IMAGEB64,
  '__module__' : 'YOLOsvc_pb2'
  # @@protoc_insertion_point(class_scope:YOLOsvc.ImageB64)
  })
_sym_db.RegisterMessage(ImageB64)


DESCRIPTOR._options = None

_YOLOSVC = _descriptor.ServiceDescriptor(
  name='YOLOsvc',
  full_name='YOLOsvc.YOLOsvc',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=294,
  serialized_end=424,
  methods=[
  _descriptor.MethodDescriptor(
    name='ObjectDetection',
    full_name='YOLOsvc.YOLOsvc.ObjectDetection',
    index=0,
    containing_service=None,
    input_type=_IMAGE,
    output_type=_DETECTIONS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='ObjectDetectionV2',
    full_name='YOLOsvc.YOLOsvc.ObjectDetectionV2',
    index=1,
    containing_service=None,
    input_type=_IMAGEB64,
    output_type=_DETECTIONS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_YOLOSVC)

DESCRIPTOR.services_by_name['YOLOsvc'] = _YOLOSVC

# @@protoc_insertion_point(module_scope)
