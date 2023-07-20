add_executable(01 01_freely_translating_rigid_body.cpp)
target_link_libraries(01 LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)


add_executable(test test_rigid_body_outline_template.cpp)
target_link_libraries(test LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)


add_executable(test_01_aosoa test_01_create_aosoa.cpp)
target_link_libraries(test_01_aosoa LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)


add_executable(test_01_rb test_01_create_rigid_body.cpp)
target_link_libraries(test_01_rb LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)

install(TARGETS 01 test_01_aosoa test_01_rb test DESTINATION ${CMAKE_INSTALL_BINDIR})
