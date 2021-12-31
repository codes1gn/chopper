add_subdirectory(chopper-compiler-shlib)
add_subdirectory(chopper-compiler-runmlir)
add_subdirectory(chopper-compiler)

# add tools
add_subdirectory(chopper-translate)
add_subdirectory(chopper-opt)

# temp workaround for compiler translate functionality, 
# use opt and wrap it into CAPI, to be replaced when 
# compiler become stable
add_subdirectory(chopper-opt-capi)
