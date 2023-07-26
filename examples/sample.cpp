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


using DataTypes = Cabana::MemberTypes<int, // ids(0)
                                      double[3], // position(1)
                                      double, // mass(2)
                                      int // body_id(3)
                                      >;

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
                                              double // mass (3)
                                              >;

using RBAoSoA = Cabana::AoSoA<RigidBodyDataType, DeviceType, VectorLength>;


auto create_aosoa_particles(int num_particles)
{
  // create rigid body (bodies) particle positions
  std::vector<double> x = {0., 1., 0., 1., 0., 1., 2., 3., 3., 4., 3., 4., 3., 4., 3., 4.};
  std::vector<double> y = {0., 0., 1., 1., -1., -1., -1., -1., 3., 3., 4., 4., 7., 7., 8., 8.};
  std::vector<double> z = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
  std::vector<int> body_id = { -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1};
  std::vector<double> m_array = { -1, -1, -1, -1, -1, -1, -1, -1, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1};
  std::vector<int> rb_limits = {8, 16};

  // sum all the number of particles and create aosoa
  AoSoAType aosoa( "particles", num_particles );

  auto positions = Cabana::slice<1>( aosoa, "positions" );
  auto ids = Cabana::slice<0>( aosoa, "ids" );
  auto m = Cabana::slice<2>( aosoa, "mass" );
  for ( std::size_t i = 0; i < aosoa.size(); ++i )
    {
      positions( i, 0 ) = x[i];
      positions( i, 1 ) = y[i];
      positions( i, 2 ) = z[i];
      ids( i ) = i;
      m( i ) = m_array[i];
    }
  return aosoa;
}


void setup_rigid_body_properties(auto aosoa, auto rb, auto index_limits){
  // auto aosoa_ids = Cabana::slice<0>          ( aosoa,    "aosoa_ids");
  auto aosoa_position = Cabana::slice<1>     ( aosoa,    "aosoa_position");
  auto aosoa_mass = Cabana::slice<2>        ( aosoa,    "aosoa_mass");
  auto aosoa_body_id = Cabana::slice<3>     ( aosoa,    "aosoa_body_id");

  auto rb_limits = Cabana::slice<1>       ( rb,      "rb_limits");
  auto rb_position = Cabana::slice<2>     ( rb,    "rb_position");
  auto rb_mass = Cabana::slice<3>        ( rb,        "rb_mass");

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

      // std::cout << "rb limits are: " << rb_limits(i, 0) << " " << rb_limits(i, 1) << "\n";

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
}

void parallel_setup_rigid_body_properties(auto aosoa, auto rb, auto index_limits){
  // auto aosoa_ids = Cabana::slice<0>          ( aosoa,    "aosoa_ids");
  auto aosoa_position = Cabana::slice<1>     ( aosoa,    "aosoa_position");
  auto aosoa_mass = Cabana::slice<2>        ( aosoa,    "aosoa_mass");
  auto aosoa_body_id = Cabana::slice<3>     ( aosoa,    "aosoa_body_id");

  auto rb_limits = Cabana::slice<1>       ( rb,      "rb_limits");
  auto rb_position = Cabana::slice<2>     ( rb,    "rb_position");
  auto rb_mass = Cabana::slice<3>        ( rb,        "rb_mass");

  /* =================================================================
     =================================================================
     ================================================================= */
  /* start: compute center of mass and total mass of the rigid body */
  /* =================================================================
     =================================================================
     ================================================================= */
  auto com_total_mass_func = KOKKOS_LAMBDA( const int i, double& total_m)
    {
      auto m_i = aosoa_mass( i );
      total_m += m_i;
    };

  /* =================================================================
     end: compute center of mass and total mass of the rigid body
     ================================================================= */
  auto parallel_compute_func = KOKKOS_LAMBDA( const int i )
    {
      // auto body_limits_i = rb_limits(i);

      double total_mass = 0.0;
      Kokkos::RangePolicy<ExecutionSpace> policy( rb_limits(i, 0), rb_limits(i, 1) );
      Kokkos::parallel_reduce(
                              "total_mass_computation", policy,
                              com_total_mass_func, total_mass);

      rb_mass(i) = total_mass;
    };

  Kokkos::RangePolicy<ExecutionSpace> policy( 0, rb_position.size() );
  Kokkos::parallel_for( "CabanaRB::RBSetup::total_mass", policy,
                        parallel_compute_func );

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

    // setup_rigid_body_properties(aosoa, rb, rigid_limits);
    parallel_setup_rigid_body_properties(aosoa, rb, rigid_limits);
  }

  // initialize the rigid body properties
  auto rb_position = Cabana::slice<2>     ( rb,    "rb_position");
  auto rb_mass = Cabana::slice<3>     ( rb,    "rb_velocity");
  auto aosoa_mass = Cabana::slice<2>     ( aosoa,    "aosoa_mass");

  std::cout << "individual mass is: " << "\n";
  for ( std::size_t i = 8; i < aosoa_mass.size(); ++i )
    {
      std::cout << aosoa_mass ( i ) << ", ";
    }

  for ( std::size_t i = 0; i < rb_position.size(); ++i )
    {
      std::cout << "\n";
      std::cout << "\n";
      std::cout << "mass of " << i << " is: " << rb_mass ( i ) << "\n";
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
