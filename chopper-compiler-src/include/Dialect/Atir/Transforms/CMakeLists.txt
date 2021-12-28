set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CHOPPERAtirPassIncGen)

add_mlir_doc(Passes CHOPPERAtirTransforms ./ -gen-pass-doc)

