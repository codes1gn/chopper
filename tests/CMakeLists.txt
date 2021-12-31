configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CHOPPER_TEST_DEPENDS
        FileCheck count not
        chopper-opt
        chopper-translate
        chopper-compiler-runmlir
        chopper-compiler-runmlir-capi
        chopper_compiler
        )

add_lit_testsuite(check-chopper "Running the chopper regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${CHOPPER_TEST_DEPENDS}
        )
set_target_properties(check-chopper PROPERTIES FOLDER "Tests")

add_lit_testsuites(CHOPPER ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${CHOPPER_TEST_DEPENDS})
