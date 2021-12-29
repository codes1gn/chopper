set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls ${CHOPPER_TABLEGEN_ARGS})
add_public_tablegen_target(CHOPPERConversionPassIncGen)

add_mlir_doc(Passes CHOPPERConversionPasses ./ -gen-pass-doc)
