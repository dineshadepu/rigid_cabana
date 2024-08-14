add_executable(01 01_freely_translating_rigid_body.cpp)
target_link_libraries(01 LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)

add_executable(02 02_rotating_rb.cpp)
target_link_libraries(02 LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)

add_executable(03 03_colliding_rigid_bodies.cpp)
target_link_libraries(03 LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)

add_executable(test test_rigid_body_outline_template.cpp)
target_link_libraries(test LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)

add_executable(test_01_aosoa test_01_create_aosoa.cpp)
target_link_libraries(test_01_aosoa LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)

add_executable(test_01_rb test_01_create_rigid_body.cpp)
target_link_libraries(test_01_rb LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)

add_executable(test_neighbours test_neighbours.cpp)
target_link_libraries(test_neighbours LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)

add_executable(test_effective_torque test_compute_effective_force_and_torque_on_rigid_body.cpp)
target_link_libraries(test_effective_torque LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)

add_executable(dem dem.cpp)
target_link_libraries(dem LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)

install(TARGETS 01 02 03 test_01_aosoa test_01_rb test test_neighbours test_effective_torque DESTINATION ${CMAKE_INSTALL_BINDIR})
