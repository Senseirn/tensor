# tensor

A header-only N-d tensor container library for C++.

## prerequisites

- C++11 (or later) Compiler
- x86-64 only

## macros

You can use macros below to specify its behavior.

| Macro Name                   | Description                                                                 | Default Value | 
| ---------------------------- | --------------------------------------------------------------------------- | ------------- | 
| TENSOR_ENABLE_ASSERTS        | if defined, runtime assertions are enabled (may cause few performance loss) | not defined   | 
| TENSOR_DEFAULT_INTERNAL_TYPE | specify the type used for index variables.                                  | int32_t       | 
| TENSOR_NAMESPACE_NAME        | specify the name of namespace used for the library.                         | ssrn          | 
| TENSOR_ENABLE_SIMD           | if defined, avx is used for optimizatoin.                                   | not deifned   | 