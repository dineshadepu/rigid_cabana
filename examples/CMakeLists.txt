add_executable(01 01_freely_translating_rigid_body.cpp)
target_link_libraries(01 LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)


add_executable(test_01 test_rigid_body_outline_template.cpp)
target_link_libraries(test_01 LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)

install(TARGETS 01 test_01 DESTINATION ${CMAKE_INSTALL_BINDIR})
