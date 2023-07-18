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

//---------------------------------------------------------------------------//
// HDF5 output example
//---------------------------------------------------------------------------//
void rigid_body_run()
{
    /*
      HDF5 is a parallel file format for large datasets. In this example, we
      will illustrate the process of storing a list of particles with
      properties, such as position, velocity, mass, radius, etc., in an HDF5
      file format.
    */

    /*
       Get parameters from the communicator. We will use MPI_COMM_WORLD for
       this example but any MPI communicator may be used.
    */
    int comm_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    if ( comm_rank == 0 )
        std::cout << "Cabana HDF5 output example\n" << std::endl;

    /*
      Start by declaring the types the particles will store. The first element
      will represent the coordinates, the second will be the particle's ID, the
      third velocity, and the fourth the radius of the particle.
    */
    using DataTypes = Cabana::MemberTypes<double[3], int, double[3], double, double, int>;

    /*
      Next declare the data layout of the AoSoA. We use the host space here
      for the purposes of this example but all memory spaces, vector lengths,
      and member type configurations are compatible.
    */
    const int VectorLength = 8;
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    /*
      Create the AoSoA.
    */
    int num_particles = 10;
    Cabana::AoSoA<DataTypes, DeviceType, VectorLength> aosoa( "A",
                                                              num_particles );

    /*
      Get the particle ids, coordinates, velocity, radius
    */
    auto positions = Cabana::slice<0>( aosoa, "positions" );
    auto ids = Cabana::slice<1>( aosoa, "ids" );
    auto velocity = Cabana::slice<2>( aosoa, "velocity" );
    auto radius = Cabana::slice<3>( aosoa, "radius" );
    auto m = Cabana::slice<4>( aosoa, "mass" );
    auto body_id = Cabana::slice<5>( aosoa, "body_id" );

    // initialize the particle properties
    for ( std::size_t i = 0; i < aosoa.size(); ++i )
    {
        // set ids of the particles
        ids( i ) = i;

        // set positions of the particles
        positions( i, 0 ) = 2. * i;
        positions( i, 1 ) = 0.;
        positions( i, 2 ) = 0.;

        // set the velocity of each particle
        velocity( i, 0 ) = 1.;
        velocity( i, 1 ) = 0.;
        velocity( i, 2 ) = 0.;

        // set the radius of each particle
        radius( i ) = 0.1;

        // set the mass of each particle
        m( i ) = 0.1;

        // body id
        body_id ( i ) = 0;
    }
    std::array<int, 2> limits = {0, 9};

    // rigid body data type
    // cm, vcm, mass, force, total no of bodies
    using RigidBodyDataType = Cabana::MemberTypes<double[3], double[3],
                                                  double, double[3], int>;

    /*
      Create the rigid body data type.
    */
    int num_bodies = 1;
    Cabana::AoSoA<RigidBodyDataType, DeviceType, VectorLength> rb( "rb",
                                                              num_bodies );

    /*
      Get the particle ids, coordinates, velocity, radius
    */
    auto cm = Cabana::slice<0>( rb, "cm" );
    auto vcm = Cabana::slice<1>( rb, "vcm" );
    auto mass = Cabana::slice<2>( rb, "mass" );
    auto force = Cabana::slice<3>( rb, "force" );
    auto nb = Cabana::slice<3>( rb, "total_no_bodies" );

    // // initialize the rigid body properties
    // for ( std::size_t i = 0; i < num_bodies; ++i )
    //   {
    //     // limits of the indices of the particles belonging to the first body
    //     // auto limit = limits[i];
    //     auto total_mass = 0.;
    //     for ( std::size_t j = limits[0]; j < limits[1]; ++j )
    //       {
    //         total_mass += m(j);
    //       }
    //     mass(i) = total_mass;
    //   }

    // // initialize the rigid body properties
    // for ( std::size_t i = 0; i < num_bodies; ++i )
    //   {
    //     std::cout << "total mass: " << mass(i) << "\n";
    //   }


    // Execute a reduce equation to compute the force on the rigid body from particles
    auto total_mass_reduce = KOKKOS_LAMBDA( const int i, double& total_m )
      {
        total_m += m( i );
      };

    using exec_space = ExecutionSpace;
    double total_mass = 0.0;
    Kokkos::RangePolicy<exec_space> policy( limits[0], limits[1] );
    Kokkos::parallel_reduce(
                            "total_mass_computation", policy,
                            total_mass_reduce, total_mass );

    mass(0) = total_mass;
    // initialize the rigid body properties
    for ( std::size_t i = 0; i < num_bodies; ++i )
      {
        std::cout << "total mass: " << mass(i) << "\n";
      }
}

int main( int argc, char* argv[] )
{

    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    rigid_body_run();

    Kokkos::finalize();

    MPI_Finalize();
    return 0;
}
