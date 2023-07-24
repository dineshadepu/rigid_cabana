/****************************************************************************
 * Copyright (c) 2018-2022 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cabana_Core.hpp>

#include <iostream>


/*
  Start by declaring the types the particles will store. The first element
  will represent the coordinates, the second will be the particle's ID, the
  third velocity, and the fourth the radius of the particle.
*/

using DataTypes = Cabana::MemberTypes<int, // ids(0)
                                      double[3], // position(1)
                                      double[3], // velocity(2)
                                      double[3], // acceleration(3)
                                      double, // mass(4)
                                      double, // density(5)
                                      double, // h (smoothing length) (6)
                                      double, // pressure (7)
                                      int, // is_fluid(8)
                                      int, // is_boundary(9)
                                      // rigid body specific properties
                                      int, // is_rb(10)
                                      double, // radius(11)
                                      int, // body_id(12)
                                      double[3], // body frame positions (13)
                                      double[3] // force on particles (DEM) (14)
                                      >;
// auto aosoa_ids = Cabana::slice<0>          ( aosoa,    "aosoa_ids")
// auto aosoa_position = Cabana::slice<1>     ( aosoa,    "aosoa_position")
// auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "aosoa_velocity")
// auto aosoa_acc = Cabana::slice<3>          ( aosoa,    "aosoa_acc")
// auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "aosoa_mass")
// auto aosoa_density = Cabana::slice<5>     ( aosoa,    "aosoa_density")
// auto aosoa_h = Cabana::slice<6>           ( aosoa,    "aosoa_h")
// auto aosoa_p = Cabana::slice<7>           ( aosoa,    "aosoa_p")
// auto aosoa_is_fluid = Cabana::slice<8>    ( aosoa,    "aosoa_is_fluid")
// auto aosoa_is_boundary = Cabana::slice<9>    ( aosoa,    "aosoa_is_boundary")
// auto aosoa_is_rb = Cabana::slice<10>       ( aosoa,    "aosoa_is_rb")
// auto aosoa_radius = Cabana::slice<11>      ( aosoa,    "aosoa_radius")
// auto aosoa_body_id = Cabana::slice<12>     ( aosoa,    "aosoa_body_id")
// auto aosoa_dx0 = Cabana::slice<13>         ( aosoa,    "aosoa_dx0")
// auto aosoa_frc_dem = Cabana::slice<14>     ( aosoa,    "aosoa_frc_dem")

/*
  Next declare the data layout of the AoSoA. We use the host space here
  for the purposes of this example but all memory spaces, vector lengths,
  and member type configurations are compatible.
*/
const int VectorLength = 8;
using MemorySpace = Kokkos::HostSpace;
using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
using AoSoAType = Cabana::AoSoA<DataTypes, DeviceType, VectorLength>;

// rigid body data type
// cm, vcm, mass, force, indices limits
using RigidBodyDataType = Cabana::MemberTypes<int, // ids (0)
                                              int[2], // limits(1)
                                              double[3], // position(2)
                                              double[3], // velocity(3)
                                              double[3], // force(4)
                                              double[3], // torque(5)
                                              double[3], // linear acceleration (6)
                                              double[3], // angular acceleration (7)
                                              double[3], // angular momentum (8)
                                              double[3], // angular velocity (9)
                                              double[3][3], // Rotation matrix (10)
                                              double, // mass (11)
                                              double, // density (12)
                                              double[3][3], // moment of inertia body frame (13)
                                              double[3][3], // moment of inertia inverse body frame (14)
                                              double[3][3], // moment of inertia global frame (15)
                                              double[3][3] // moment of inertia inverse global frame (16)
                                              >;

// auto rb_ids = Cabana::slice<0>          ( aosoa,         "rb_ids")
// auto rb_limits = Cabana::slice<1>       ( aosoa,      "rb_limits")
// auto rb_position = Cabana::slice<2>     ( aosoa,    "rb_position")
// auto rb_velocity = Cabana::slice<3>     ( aosoa,    "rb_velocity")
// auto rb_force = Cabana::slice<4>        ( aosoa,       "rb_force")
// auto rb_torque = Cabana::slice<5>       ( aosoa,      "rb_torque")
// auto rb_lin_acc = Cabana::slice<6>      ( aosoa,     "rb_lin_acc")
// auto rb_ang_acc = Cabana::slice<7>      ( aosoa,     "rb_ang_acc")
// auto rb_ang_mom = Cabana::slice<8>      ( aosoa,     "rb_ang_mom")
// auto rb_ang_vel = Cabana::slice<9>      ( aosoa,     "rb_ang_vel")
// auto rb_rot_mat = Cabana::slice<10>     ( aosoa,     "rb_rot_mat")
// auto rb_mass = Cabana::slice<11>        ( aosoa,        "rb_mass")
// auto rb_density = Cabana::slice<12>     ( aosoa,     "rb_density")
// auto rb_body_moi = Cabana::slice<13>    ( aosoa,    "rb_body_moi")
// auto rb_inv_body_moi = Cabana::slice<14>( aosoa,    "rb_inv_body_moi")
// auto rb_global_moi = Cabana::slice<15>  ( aosoa,    "rb_global_moi")
// auto rb_inv_global_moi = Cabana::slice<16>( aosoa,    "rb_inv_global_moi")


  // auto aosoa_ids = Cabana::slice<0>          ( aosoa,    "aosoa_ids");
  // auto aosoa_position = Cabana::slice<1>     ( aosoa,    "aosoa_position");
  // auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "aosoa_velocity");
  // auto aosoa_acc = Cabana::slice<3>          ( aosoa,    "aosoa_acc");
  // auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "aosoa_mass");
  // auto aosoa_density = Cabana::slice<5>     ( aosoa,    "aosoa_density");
  // auto aosoa_h = Cabana::slice<6>           ( aosoa,    "aosoa_h");
  // auto aosoa_p = Cabana::slice<7>           ( aosoa,    "aosoa_p");
  // auto aosoa_is_fluid = Cabana::slice<8>    ( aosoa,    "aosoa_is_fluid");
  // auto aosoa_is_boundary = Cabana::slice<9>    ( aosoa,    "aosoa_is_boundary");
  // auto aosoa_is_rb = Cabana::slice<10>       ( aosoa,    "aosoa_is_rb");
  // auto aosoa_radius = Cabana::slice<11>      ( aosoa,    "aosoa_radius");
  // auto aosoa_body_id = Cabana::slice<12>     ( aosoa,    "aosoa_body_id");
  // auto aosoa_dx0 = Cabana::slice<13>         ( aosoa,    "aosoa_dx0");
  // auto aosoa_frc_dem = Cabana::slice<14>     ( aosoa,    "aosoa_frc_dem");

  // auto rb_ids = Cabana::slice<0>          ( rb,         "rb_ids");
  // auto rb_limits = Cabana::slice<1>       ( rb,      "rb_limits");
  // auto rb_position = Cabana::slice<2>     ( rb,    "rb_position");
  // auto rb_velocity = Cabana::slice<3>     ( rb,    "rb_velocity");
  // auto rb_force = Cabana::slice<4>        ( rb,       "rb_force");
  // auto rb_torque = Cabana::slice<5>       ( rb,      "rb_torque");
  // auto rb_lin_acc = Cabana::slice<6>      ( rb,     "rb_lin_acc");
  // auto rb_ang_acc = Cabana::slice<7>      ( rb,     "rb_ang_acc");
  // auto rb_ang_mom = Cabana::slice<8>      ( rb,     "rb_ang_mom");
  // auto rb_ang_vel = Cabana::slice<9>      ( rb,     "rb_ang_vel");
  // auto rb_rot_mat = Cabana::slice<10>     ( rb,     "rb_rot_mat");
  // auto rb_mass = Cabana::slice<11>        ( rb,        "rb_mass");
  // auto rb_density = Cabana::slice<12>     ( rb,     "rb_density");
  // auto rb_body_moi = Cabana::slice<13>    ( rb,    "rb_body_moi");
  // auto rb_inv_body_moi = Cabana::slice<14>( rb,    "rb_inv_body_moi");
  // auto rb_global_moi = Cabana::slice<15>  ( rb,    "rb_global_moi");
  // auto rb_inv_global_moi = Cabana::slice<16>( rb,    "rb_inv_global_moi");

using RBAoSoA = Cabana::AoSoA<RigidBodyDataType, DeviceType, VectorLength>;

void set_fluid_properties_of_aosoa(AoSoAType aosoa, std::vector<double>& xf, std::vector<double>& yf, std::vector<double>& zf, std::vector<int>& limits)
{
  /*
    Get the particle ids, coordinates, velocity, radius
  */
  auto positions = Cabana::slice<1>( aosoa, "positions" );
  auto velocity = Cabana::slice<2>( aosoa, "velocity" );

  // initialize the particle properties
  int local_index = 0;
  for ( std::size_t i = limits[0]; i < limits[1]; ++i )
    {
      // set positions of the particles
      positions( i, 0 ) = xf[local_index];
      positions( i, 1 ) = yf[local_index];
      positions( i, 2 ) = zf[local_index];

      // set the velocity of each particle
      velocity( i, 0 ) = 0.;
      velocity( i, 1 ) = 0.;
      velocity( i, 2 ) = 0.;
      local_index += 1;
    }
  // std::array<int, 2> limits = {0, 9};
}


auto create_aosoa_particles(int num_particles)
{

  // create fluid particle positions
  std::vector<double> xf = {0., 1., 0., 1.};
  std::vector<double> yf = {0., 0., 1., 1.};
  std::vector<double> zf = {0., 0., 0., 0.};
  std::vector<int> fluid_limits = {0, 4};

  // create boundary particle positions
  std::vector<double> xb = {0., 1., 2., 3.};
  std::vector<double> yb = {-1., -1., -1., -1.};
  std::vector<double> zb = {0., 0., 0., 0.};
  std::vector<int> boundary_limits = {4, 8};

  // create rigid body (bodies) particle positions
  std::vector<double> x_rb = {3., 4., 3., 4., 3., 4., 3., 4.};
  std::vector<double> y_rb = {3., 3., 4., 4., 7., 7., 8., 8.};
  std::vector<double> z_rb = {0., 0., 0., 0., 0., 0., 0., 0.};
  std::vector<int> body_id = { -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1};
  std::vector<double> m_array = { -1, -1, -1, -1, -1, -1, -1, -1, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1};
  std::vector<int> rb_limits = {8, 16};

  // sum all the number of particles and create aosoa
  AoSoAType aosoa( "particles", num_particles );

  auto ids = Cabana::slice<0>( aosoa, "ids" );
  auto m = Cabana::slice<4>( aosoa, "mass" );
  for ( std::size_t i = 0; i < aosoa.size(); ++i )
    {
      ids( i ) = i;
      m( i ) = m_array[i];
    }

  // setup the fluid positions and properties of the aosoa
  set_fluid_properties_of_aosoa(aosoa, xf, yf, zf, fluid_limits);

  // setup the boundary positions and properties of the aosoa
  set_fluid_properties_of_aosoa(aosoa, xb, yb, zf, boundary_limits);

  // setup the rigid body positions and properties of the aosoa
  set_fluid_properties_of_aosoa(aosoa, x_rb, y_rb, z_rb, rb_limits);

  // set the body id for the particles
  auto body_ids = Cabana::slice<12>( aosoa, "body_id" );
  // initialize the rigid body properties
  for ( std::size_t i = 0; i < num_particles; ++i )
    {
      body_ids(i) = body_id[i];
    }
  return aosoa;
}


void setup_rigid_body_properties(auto aosoa, auto rb, auto index_limits){
  // auto aosoa_ids = Cabana::slice<0>          ( aosoa,    "aosoa_ids");
  auto aosoa_position = Cabana::slice<1>     ( aosoa,    "aosoa_position");
  auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "aosoa_mass");
  auto aosoa_body_id = Cabana::slice<12>     ( aosoa,    "aosoa_body_id");
  auto aosoa_dx0 = Cabana::slice<13>         ( aosoa,    "aosoa_dx0");

  auto rb_limits = Cabana::slice<1>       ( rb,      "rb_limits");
  auto rb_position = Cabana::slice<2>     ( rb,    "rb_position");
  auto rb_mass = Cabana::slice<11>        ( rb,        "rb_mass");

  /* =================================================================
     =================================================================
     ================================================================= */
  /* start: compute center of mass and total mass of the rigid body */
  /* =================================================================
     =================================================================
     ================================================================= */
  auto com_total_mass_func = KOKKOS_LAMBDA( const int i, double& total_m, double& cm_x, double& cm_y, double& cm_z)
    {
      auto m_i = aosoa_mass( i );
      total_m += m_i;
      cm_x += m_i * aosoa_position(i, 0);
      cm_y += m_i * aosoa_position(i, 1);
      cm_z += m_i * aosoa_position(i, 2);
    };

  for ( std::size_t i = 0; i < rb_position.size(); ++i )
    {
      double total_mass = 0.0;
      double cm_x = 0.0;
      double cm_y = 0.0;
      double cm_z = 0.0;

      std::cout << "rb limits are: " << rb_limits(i, 0) << " " << rb_limits(i, 1) << "\n";

      Kokkos::RangePolicy<ExecutionSpace> policy( rb_limits(i, 0), rb_limits(i, 1) );
      Kokkos::parallel_reduce(
                              "COM_and_total_mass_computation", policy,
                              com_total_mass_func, total_mass, cm_x, cm_y, cm_z);
      rb_mass(i) = total_mass;
      rb_position(i, 0) = cm_x / total_mass;
      rb_position(i, 1) = cm_y / total_mass;
      rb_position(i, 2) = cm_z / total_mass;
    }
  /* =================================================================
     end: compute center of mass and total mass of the rigid body
     ================================================================= */

  /* =================================================================
     start: save the initial positions of the particles in body frame axis
     ================================================================= */
  auto rb_save_initial_positions_lambda_func = KOKKOS_LAMBDA( const int i )
    {
      auto particle_body_id = aosoa_body_id(i);
      aosoa_dx0(i, 0) = aosoa_position(i, 0) - rb_position( particle_body_id, 0 );
      aosoa_dx0(i, 1) = aosoa_position(i, 1) - rb_position( particle_body_id, 1 );
      aosoa_dx0(i, 2) = aosoa_position(i, 2) - rb_position( particle_body_id, 2 );
    };
  Kokkos::RangePolicy<ExecutionSpace> policy( index_limits[0], index_limits[1] );
  Kokkos::parallel_for( "CabanaRB:RBSetup:SaveInitPos", policy,
                        rb_save_initial_positions_lambda_func );
  /* =================================================================
     end: save the initial positions of the particles in body frame axis
     ================================================================= */
}

void set_linear_velocity_rigid_body(auto aosoa, auto rb, auto body_no, std::vector<double>& lin_vel){
  // get the properties of the particles and rigid body
  auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "aosoa_velocity");
  auto rb_limits = Cabana::slice<1>       ( rb,      "rb_limits");
  auto rb_velocity = Cabana::slice<3>     ( rb,    "rb_velocity");

  // set the linear velocity of the rigid body
  rb_velocity( body_no, 0) = lin_vel[0];
  rb_velocity( body_no, 1) = lin_vel[1];
  rb_velocity( body_no, 2) = lin_vel[2];

  // set the velocity of the particles of the rigid body
  auto rb_set_lin_vel_func = KOKKOS_LAMBDA( const int i )
    {

      aosoa_velocity( i, 0) = rb_velocity( body_no, 0);
      aosoa_velocity( i, 1) = rb_velocity( body_no, 1);
      aosoa_velocity( i, 2) = rb_velocity( body_no, 2);

    };
  // range of particle indices for the current body no is
  auto start = rb_limits(body_no, 0);
  auto end = rb_limits(body_no, 1);
  Kokkos::RangePolicy<ExecutionSpace> policy( start, end );
  Kokkos::parallel_for( "CabanaRB:RB:SetLinVel", policy,
                        rb_set_lin_vel_func );
}


void rigid_body_gtvf_stage_1(auto rb, double dt){
  auto rb_velocity = Cabana::slice<3>     ( rb,    "rb_velocity");
  auto rb_force = Cabana::slice<4>        ( rb,       "rb_force");
  auto rb_lin_acc = Cabana::slice<6>      ( rb,     "rb_lin_acc");
  auto rb_mass = Cabana::slice<11>        ( rb,        "rb_mass");

  auto half_dt = dt * 0.5;
  auto rb_gtvf_stage_1_lambda_func = KOKKOS_LAMBDA( const int i )
    {
      auto mass_i = rb_mass( i );
      auto mass_i_1 = 1. / mass_i;
      rb_lin_acc( i, 0 ) = rb_force( i, 0 ) * mass_i_1;
      rb_lin_acc( i, 1 ) = rb_force( i, 1 ) * mass_i_1;
      rb_lin_acc( i, 2 ) = rb_force( i, 2 ) * mass_i_1;

      rb_velocity( i, 0 ) += rb_lin_acc( i, 0 ) * half_dt;
      rb_velocity( i, 1 ) += rb_lin_acc( i, 1 ) * half_dt;
      rb_velocity( i, 2 ) += rb_lin_acc( i, 2 ) * half_dt;
    };
  Kokkos::RangePolicy<ExecutionSpace> policy( 0, rb_velocity.size() );
  Kokkos::parallel_for( "CabanaRB:Integrator:RBGTVFStage1", policy,
                        rb_gtvf_stage_1_lambda_func );
}


void rigid_body_particles_gtvf_stage_1(auto aosoa, auto rb, auto dt, auto index_limits){
  auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "aosoa_velocity");
  auto aosoa_body_id = Cabana::slice<12>     ( aosoa,    "aosoa_body_id");
  auto rb_velocity = Cabana::slice<3>     ( rb,    "rb_velocity");

  auto half_dt = dt * 0.5;
  auto rb_particles_gtvf_stage_1_lambda_func = KOKKOS_LAMBDA( const int i )
    {
      auto particle_body_id = aosoa_body_id(i);
      aosoa_velocity( i, 0 ) = rb_velocity( particle_body_id, 0 );
      aosoa_velocity( i, 1 ) = rb_velocity( particle_body_id, 1 );
      aosoa_velocity( i, 2 ) = rb_velocity( particle_body_id, 2 );
    };

  Kokkos::RangePolicy<ExecutionSpace> policy( index_limits[0], index_limits[1] );
  Kokkos::parallel_for( "CabanaRB:Integrator:RBParticlesGTVFStage1", policy,
                        rb_particles_gtvf_stage_1_lambda_func );
}


void rigid_body_gtvf_stage_2(auto rb, auto dt){
  auto rb_position = Cabana::slice<2>     ( rb,    "rb_position");
  auto rb_velocity = Cabana::slice<3>     ( rb,    "rb_velocity");

  auto rb_gtvf_stage_2_lambda_func = KOKKOS_LAMBDA( const int i )
    {
      rb_position( i, 0 ) += rb_velocity( i, 0 ) * dt;
      rb_position( i, 1 ) += rb_velocity( i, 1 ) * dt;
      rb_position( i, 2 ) += rb_velocity( i, 2 ) * dt;
    };
  Kokkos::RangePolicy<ExecutionSpace> policy( 0, rb_velocity.size() );
  Kokkos::parallel_for( "CabanaRB:Integrator:RBGTVFStage2", policy,
                        rb_gtvf_stage_2_lambda_func );
}


void rigid_body_particles_gtvf_stage_2(auto aosoa, auto rb, auto dt, auto index_limits){
  auto aosoa_position = Cabana::slice<1>     ( aosoa,    "aosoa_velocity");
  auto aosoa_body_id = Cabana::slice<12>     ( aosoa,    "aosoa_body_id");
  auto aosoa_dx0 = Cabana::slice<13>         ( aosoa,    "aosoa_dx0");

  auto rb_position = Cabana::slice<2>     ( rb,    "rb_position");

  auto rb_particles_gtvf_stage_2_lambda_func = KOKKOS_LAMBDA( const int i )
    {
      auto particle_body_id = aosoa_body_id(i);
      aosoa_position( i, 0 ) = rb_position( particle_body_id, 0 ) + aosoa_dx0(i, 0);
      aosoa_position( i, 1 ) = rb_position( particle_body_id, 1 ) + aosoa_dx0(i, 1);
      aosoa_position( i, 2 ) = rb_position( particle_body_id, 2 ) + aosoa_dx0(i, 2);
    };

  Kokkos::RangePolicy<ExecutionSpace> policy( index_limits[0], index_limits[1] );
  Kokkos::parallel_for( "CabanaRB:Integrator:RBParticlesGTVFStage2", policy,
                        rb_particles_gtvf_stage_2_lambda_func );
}

void compute_force_on_rigid_body_particles(auto aosoa, auto dt){


}


void compute_effective_force_and_torque_on_rigid_body(auto rb, auto aosoa){


}


void rigid_body_gtvf_stage_3(auto rb, auto dt){
  auto rb_velocity = Cabana::slice<3>     ( rb,    "rb_velocity");
  auto rb_force = Cabana::slice<4>        ( rb,       "rb_force");
  auto rb_lin_acc = Cabana::slice<6>      ( rb,     "rb_lin_acc");
  auto rb_mass = Cabana::slice<11>        ( rb,        "rb_mass");

  auto half_dt = dt * 0.5;
  auto rb_gtvf_stage_3_lambda_func = KOKKOS_LAMBDA( const int i )
    {
      auto mass_i = rb_mass( i );
      auto mass_i_1 = 1. / mass_i;
      rb_lin_acc( i, 0 ) = rb_force( i, 0 ) * mass_i_1;
      rb_lin_acc( i, 1 ) = rb_force( i, 1 ) * mass_i_1;
      rb_lin_acc( i, 2 ) = rb_force( i, 2 ) * mass_i_1;

      rb_velocity( i, 0 ) += rb_lin_acc( i, 0 ) * half_dt;
      rb_velocity( i, 1 ) += rb_lin_acc( i, 1 ) * half_dt;
      rb_velocity( i, 2 ) += rb_lin_acc( i, 2 ) * half_dt;
    };
  Kokkos::RangePolicy<ExecutionSpace> policy( 0, rb_velocity.size() );
  Kokkos::parallel_for( "CabanaRB:Integrator:RBGTVFStage3", policy,
                        rb_gtvf_stage_3_lambda_func );
}

void rigid_body_particles_gtvf_stage_3(auto aosoa, auto rb, auto dt, auto index_limits){
  auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "aosoa_velocity");
  auto aosoa_body_id = Cabana::slice<12>     ( aosoa,    "aosoa_body_id");
  auto rb_velocity = Cabana::slice<3>     ( rb,    "rb_velocity");

  auto half_dt = dt * 0.5;
  auto rb_particles_gtvf_stage_3_lambda_func = KOKKOS_LAMBDA( const int i )
    {
      auto particle_body_id = aosoa_body_id(i);
      aosoa_velocity( i, 0 ) = rb_velocity( particle_body_id, 0 );
      aosoa_velocity( i, 1 ) = rb_velocity( particle_body_id, 1 );
      aosoa_velocity( i, 2 ) = rb_velocity( particle_body_id, 2 );
    };

  Kokkos::RangePolicy<ExecutionSpace> policy( index_limits[0], index_limits[1] );
  Kokkos::parallel_for( "CabanaRB:Integrator:RBParticlesGTVFStage1", policy,
                        rb_particles_gtvf_stage_3_lambda_func );
}


void output_data(auto aosoa, int num_particles, int step, double time)
{
  // This is for setting HDF5 options
  auto ids = Cabana::slice<0>( aosoa, "ids" );
  auto mass = Cabana::slice<4>        ( aosoa,    "mass");
  auto positions = Cabana::slice<1>( aosoa, "positions" );
  auto velocity = Cabana::slice<2>( aosoa, "velocity" );
  auto radius = Cabana::slice<11>( aosoa, "radius" );
  auto body_ids = Cabana::slice<12>( aosoa, "body_id" );

  Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
  Cabana::Experimental::HDF5ParticleOutput::
    writeTimeStep(
          h5_config, "particles", MPI_COMM_WORLD,
          step, time, num_particles, positions,
          ids, velocity, radius, body_ids, mass);
}


//---------------------------------------------------------------------------//
// TODO: explain this function in short
//---------------------------------------------------------------------------//
void run()
{
  int comm_rank = -1;
  MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

  if ( comm_rank == 0 )
    std::cout << "Cabana Rigid body solver example\n" << std::endl;

  auto num_particles = 16;
  auto aosoa = create_aosoa_particles(num_particles);
  /*
    Create the rigid body data type.
  */
  int num_bodies = 2;
  std::vector<int> rigid_limits = {8, 16};
  RBAoSoA rb( "rb", num_bodies );
  // get the rigid body limits
  {
  auto limits = Cabana::slice<1>( rb, "total_no_bodies" );
  // limits of the first body
  limits(0, 0) = 8;
  limits(0, 1) = 12;

  limits(1, 0) = 12;
  limits(1, 1) = 16;

  setup_rigid_body_properties(aosoa, rb, rigid_limits);
  }

  std::vector<double> linear_velocity_0 = {4., 4., 0.};
  set_linear_velocity_rigid_body(aosoa, rb, 0, linear_velocity_0);
  std::vector<double> linear_velocity_1 = {4., -4., 0.};
  set_linear_velocity_rigid_body(aosoa, rb, 1, linear_velocity_1);

  // output the data
  // output_data(aosoa, num_particles, 0, 0.);
  // output_data(aosoa, num_particles, 0, 0.);

  auto dt = 1e-2;
  auto final_time = 1000. * dt;
  auto time = 0.;
  int steps = final_time / dt;
  int print_freq = 30;

  // Main timestep loop
  for ( int step = 0; step < steps; step++ )
    {
      rigid_body_gtvf_stage_1(rb, dt);
      rigid_body_particles_gtvf_stage_1(aosoa, rb, dt, rigid_limits);

      rigid_body_gtvf_stage_2(rb, dt);
      rigid_body_particles_gtvf_stage_2(aosoa, rb, dt, rigid_limits);

      compute_force_on_rigid_body_particles(aosoa, dt);
      compute_effective_force_and_torque_on_rigid_body(rb, aosoa);

      rigid_body_gtvf_stage_3(rb, dt);
      rigid_body_particles_gtvf_stage_3(aosoa, rb, dt, rigid_limits);

      // // initialize the rigid body properties
      // auto rb_position = Cabana::slice<2>     ( rb,    "rb_position");
      // auto rb_velocity = Cabana::slice<3>     ( rb,    "rb_velocity");
      // for ( std::size_t i = 0; i < rb_position.size() - 1; ++i )
      //   {
      //     std::cout << "\n";
      //     std::cout << "\n";
      //     std::cout << "time is: " << time << "\n";
      //     std::cout << "rb pos x pos: " << rb_position(i, 0) << ", ";
      //     std::cout << "rb pos y pos: " << rb_position(i, 1) << ", ";
      //     std::cout << "rb pos z pos: " << rb_position(i, 2) << "\n";

      //     std::cout << "rb vel x pos: " << rb_velocity(i, 0) << ", ";
      //     std::cout << "rb vel y pos: " << rb_velocity(i, 1) << ", ";
      //     std::cout << "rb vel z pos: " << rb_velocity(i, 2) << "\n";
      //   }

      // // A configuration object is necessary for tuning HDF5 options.
      // Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;

      // // For example, MPI I/O is independent by default, but collective operations
      // // can be enabled by setting:
      // h5_config.collective = true;
      if ( step % print_freq == 0 )
        {
      output_data(aosoa, num_particles, step, time);
        }

      time += dt;

    }
}

int main( int argc, char* argv[] )
{

  MPI_Init( &argc, &argv );
  Kokkos::initialize( argc, argv );

  run();

  Kokkos::finalize();

  MPI_Finalize();
  return 0;
}
