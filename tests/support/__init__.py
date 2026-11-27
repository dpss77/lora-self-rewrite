"""Shared fakes and factories for layered tests.

Reusable boundary doubles (tokenizers, fake engines, volume stubs, row
builders) belong here so test modules do not import each other. Scenario-specific
setup stays in the layer module that owns it.
"""
