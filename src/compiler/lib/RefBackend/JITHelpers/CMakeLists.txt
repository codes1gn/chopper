add_chopper_library(CHOPPERRefBackendJITHelpers
  JITModule.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SRC_DIR}/include/chopper/RefBackend/JITHelpers

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CHOPPERRuntime
  CHOPPERRefBackend
  MLIRExecutionEngine
  )

mlir_check_all_link_libraries(CHOPPERRefBackend)
