# Domain of onnxifier

In ONNX, a domain is a way to organize operators and types. The default domain is "ai.onnx". However, there can also be custom domains to group related operators and types. For example, "ai.microsoft" or "ai.openvino".

In this sub-package, we provide a set of tools to work with custom domains. Including translation of domain-specific operators to the default domain, and conversion of models between different domains.
