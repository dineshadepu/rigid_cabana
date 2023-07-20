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
//                                    positions, ids, is_fluid, is_boundary, is_rb, velocity, radius, mass, body_id
using DataTypes = Cabana::MemberTypes<double[3], int, int, int, int, double[3], double, double, int>;

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


auto create_aosoa_particles(int num_particles)
{
  AoSoAType aosoa( "A", num_particles );

  // Cabana::AoSoA<DataTypes, DeviceType, VectorLength> aosoa( "A",
  //                            num_particles );

  /*
    Get the particle ids, coordinates, velocity, radius
  */
  // auto positions = Cabana::slice<0>( aosoa, "positions" );
  auto ids = Cabana::slice<1>( aosoa, "ids" );
  // auto is_fluid = Cabana::slice<2>( aosoa, "ids" );
  // auto is_boundary = Cabana::slice<3>( aosoa, "ids" );
  // auto is_rb = Cabana::slice<4>( aosoa, "ids" );
  // auto velocity = Cabana::slice<5>( aosoa, "velocity" );
  // auto radius = Cabana::slice<6>( aosoa, "radius" );
  // auto m = Cabana::slice<7>( aosoa, "mass" );
  // auto body_id = Cabana::slice<8>( aosoa, "body_id" );

  // initialize the particle properties
  for ( std::size_t i = 0; i < aosoa.size(); ++i )
    {
      // set ids of the particles
      ids( i ) = i;

      // // set positions of the particles
      // positions( i, 0 ) = 2. * i;
      // positions( i, 1 ) = 0.;
      // positions( i, 2 ) = 0.;

      // // set the velocity of each particle
      // velocity( i, 0 ) = 1.;
      // velocity( i, 1 ) = 0.;
      // velocity( i, 2 ) = 0.;

      // // set the radius of each particle
      // radius( i ) = 0.1;

      // // set the mass of each particle
      // m( i ) = 0.1;

      // // body id
      // body_id ( i ) = 0;
    }
  // std::array<int, 2> limits = {0, 9};
  return aosoa;
}


void set_fluid_properties_of_aosoa(AoSoAType aosoa, std::vector<double>& xf, std::vector<double>& yf, std::vector<double>& zf, std::vector<int>& limits)
{
  /*
    Get the particle ids, coordinates, velocity, radius
  */
  auto positions = Cabana::slice<0>( aosoa, "positions" );
  // auto ids = Cabana::slice<1>( aosoa, "ids" );
  auto is_fluid = Cabana::slice<2>( aosoa, "is_fluid" );
  // auto is_boundary = Cabana::slice<3>( aosoa, "ids" );
  // auto is_rb = Cabana::slice<4>( aosoa, "ids" );
  auto velocity = Cabana::slice<5>( aosoa, "velocity" );
  // auto radius = Cabana::slice<6>( aosoa, "radius" );
  // auto m = Cabana::slice<7>( aosoa, "mass" );
  // auto body_id = Cabana::slice<8>( aosoa, "body_id" );

  // initialize the particle properties
  int local_index = 0;
  for ( std::size_t i = limits[0]; i < limits[1]; ++i )
    {
      std::cout << "i is: " << i << "\n";

      // set positions of the particles
      positions( i, 0 ) = xf[local_index];
      positions( i, 1 ) = yf[local_index];
      positions( i, 2 ) = zf[local_index];

      // set the velocity of each particle
      velocity( i, 0 ) = 0.;
      velocity( i, 1 ) = 0.;
      velocity( i, 2 ) = 0.;

      // // set the radius of each particle
      // radius( i ) = 0.1;

      // // set the mass of each particle
      // m( i ) = 0.1;

      // // body id
      // body_id ( i ) = 0;

      local_index += 1;
    }
  // std::array<int, 2> limits = {0, 9};
}



//---------------------------------------------------------------------------//
// TODO: explain this functino in short
//---------------------------------------------------------------------------//
void run()
{
  int comm_rank = -1;
  MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

  if ( comm_rank == 0 )
    std::cout << "Cabana HDF5 output example\n" << std::endl;

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
  std::vector<int> rb_limits = {8, 16};

  // sum all the number of particles and create aosoa
  auto num_particles = 16;
  auto aosoa = create_aosoa_particles(num_particles);

  // setup the fluid positions and properties of the aosoa
  set_fluid_properties_of_aosoa(aosoa, xf, yf, zf, fluid_limits);

  // setup the boundary positions and properties of the aosoa
  set_fluid_properties_of_aosoa(aosoa, xb, yb, zf, boundary_limits);

  // setup the rigid body positions and properties of the aosoa
  set_fluid_properties_of_aosoa(aosoa, x_rb, y_rb, zf, rb_limits);

  // This is for setting HDF5 options
  auto ids = Cabana::slice<1>( aosoa, "ids" );
  auto positions = Cabana::slice<0>( aosoa, "positions" );
  auto velocity = Cabana::slice<5>( aosoa, "velocity" );
  auto radius = Cabana::slice<6>( aosoa, "radius" );

  // initialize the rigid body properties
  for ( std::size_t i = 0; i < num_particles; ++i )
    {
      std::cout << "x pos: " << positions(i, 0) << ", ";
      std::cout << "y pos: " << positions(i, 1) << ", ";
      std::cout << "z pos: " << positions(i, 2) << "\n";
    }
  Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
  Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
                              h5_config, "particles", MPI_COMM_WORLD, 0, 0.1, num_particles, positions,
                              ids, velocity, radius);

  Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
                              h5_config, "particles", MPI_COMM_WORLD, 100, 0.1, num_particles, positions,
                              ids, velocity, radius);

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
