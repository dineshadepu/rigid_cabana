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


const int VectorLength = 8;
using MemorySpace = Kokkos::HostSpace;
using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

using ListAlgorithm = Cabana::FullNeighborTag;
using ListType =
  Cabana::VerletList<MemorySpace, ListAlgorithm, Cabana::VerletLayout2D>;

// void output_data(auto aosoa, int num_particles, int step, double time)
// {
//   // This is for setting HDF5 options
//   auto ids = Cabana::slice<0>( aosoa, "ids" );
//   // auto mass = Cabana::slice<4>( aosoa,    "mass");
//   auto positions = Cabana::slice<1>( aosoa, "positions" );
//   // auto velocity = Cabana::slice<2>( aosoa, "velocity" );
//   // auto radius = Cabana::slice<11>( aosoa, "radius" );
//   // auto body_ids = Cabana::slice<12>( aosoa, "body_id" );

//   Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
//   Cabana::Experimental::HDF5ParticleOutput::
//     writeTimeStep(
//           h5_config, "particles", MPI_COMM_WORLD,
//           step, time, num_particles, positions,
//           ids, velocity, radius, body_ids, mass);
// }


//---------------------------------------------------------------------------//
// TODO: explain this function in short
//---------------------------------------------------------------------------//
void run()
{
  int comm_rank = -1;
  MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

  if ( comm_rank == 0 )
    std::cout << "Cabana Rigid body solver example\n" << std::endl;


  using DataTypes = Cabana::MemberTypes<double[3], int>;
  // using DataTypes = Cabana::MemberTypes<int, double[3]>;



  // using DataTypes = Cabana::MemberTypes<int, // ids(0)
  //                                       double[3], // position(1)
  //                                       double[3], // velocity(2)
  //                                       double[3], // acceleration(3)
  //                                       double, // mass(4)
  //                                       double, // density(5)
  //                                       double, // h (smoothing length) (6)
  //                                       double, // pressure (7)
  //                                       int, // is_fluid(8)
  //                                       int, // is_boundary(9)
  //                                       // rigid body specific properties
  //                                       int, // is_rb(10)
  //                                       double, // radius(11)
  //                                       int, // body_id(12)
  //                                       double[3], // body frame positions (13)
  //                                       double[3] // force on particles (DEM) (14)
  //                                       >;

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

  auto num_particles = 16;

  std::vector<double> x_rb = {0., 1., 0., 1., 0., 1., 2., 3., 3., 4., 3., 4., 3., 4., 3., 4.};
  std::vector<double> y_rb = {0., 0., 1., 1., -1., -1., -1., -1., 3., 3., 4., 4., 7., 7., 8., 8.};
  std::vector<double> z_rb = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
  std::vector<int> is_rb = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> body_id = { -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1};
  std::vector<double> m_array = { -1, -1, -1, -1, -1, -1, -1, -1, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1};
  std::vector<int> rb_limits = {8, 16};

  Cabana::AoSoA<DataTypes, DeviceType, VectorLength> aosoa( "A", num_particles );
  // sum all the number of particles and create aosoa
  // AoSoAType aosoa( "particles", num_particles );

  auto ids = Cabana::slice<1>( aosoa, "ids" );
  auto position = Cabana::slice<0>( aosoa, "position" );
  // auto m = Cabana::slice<4>( aosoa, "mass" );
  for ( std::size_t i = 0; i < aosoa.size(); ++i )
    {
      ids( i ) = i;
      // m( i ) = m_array[i];
      position( i, 0 ) = x_rb[i];
      position( i, 1 ) = y_rb[i];
      position( i, 2 ) = z_rb[i];
    }

  // output_data(aosoa, num_particles, 0, 0.);
  // ================================================
  // ================================================
  // create the neighbor list
  // ================================================
  // ================================================
  double grid_min[3] = { 0.0, -2.0, -1.0 };
  double grid_max[3] = { 25.0, 6.0, 1.0 };
  // double neighborhood_radius = 1.3;
  // double cell_ratio = 1.0;

  double neighborhood_radius = 0.25;
  double cell_ratio = 1.0;
  using ListAlgorithm = Cabana::FullNeighborTag;
  using ListType =
    Cabana::VerletList<MemorySpace, ListAlgorithm, Cabana::VerletLayoutCSR,
                       Cabana::TeamOpTag>;
  ListType verlet_list( position, 0, position.size(), neighborhood_radius,
                        cell_ratio, grid_min, grid_max );

  // ListType verlet_list( positions, 0,
  //                       positions.size(), neighborhood_radius,
  //                       cell_ratio, grid_min, grid_max );
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
