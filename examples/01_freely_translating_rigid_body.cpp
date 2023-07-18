/****************************************************************************
 * Copyright (c) 2022 by Oak Ridge National Laboratory                      *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of CabanaPD. CabanaPD is distributed under a           *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <fstream>
#include <iostream>

#include <Kokkos_Core.hpp>
#include <Cabana_Core.hpp>


void freely_moving_rigid_body()
{
    /*
      Given a list of particle positions, for every particle in the list a
      Verlet list computes the other particles in the list that are within
      some specified cutoff distance from the particle. Once created, the
      Verlet list data can be accessed with the neighbor list interface. We
      will demonstrate building a Verlet list and accessing it's data in
      this example.
    */

    // std::cout << "Cabana Verlet Neighbor List Example\n" << std::endl;

    /*
       Start by declaring the types in our tuples will store. The first
       member will be the coordinates, the second an id.
    */
    using DataTypes = Cabana::MemberTypes<double[3], int>;

    /*
      Next declare the data layout of the AoSoA. We use the host space here
      for the purposes of this example but all memory spaces, vector lengths,
      and member type configurations are compatible with neighbor lists.
    */
    const int VectorLength = 8;
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    /*
       Create the AoSoA.
    */
    int num_tuple = 81;
    Cabana::AoSoA<DataTypes, DeviceType, VectorLength> aosoa( "A", num_tuple );

    /*
      Define the parameters of the Cartesian grid over which we will build the
      particles. This is a simple 3x3x3 uniform grid on [0,3] in each
      direction. Each grid cell has a size of 1 in each dimension.
     */
    double grid_min[3] = { 0.0, 0.0, 0.0 };
    double grid_max[3] = { 3.0, 3.0, 3.0 };
    double grid_delta[3] = { 1.0, 1.0, 1.0 };

    /*
      Create the particle ids.
    */
    auto ids = Cabana::slice<1>( aosoa );
    for ( std::size_t i = 0; i < aosoa.size(); ++i )
        ids( i ) = i;

    /*
      Create the particle coordinates. We will put 3 particles in the center
      of each cell. We will set the Verlet list parameters such that each
      particle should only neighbor the other particles it shares a cell with.
    */
    auto positions = Cabana::slice<0>( aosoa );
    int ppc = 3;
    int particle_counter = 0;
    for ( int p = 0; p < ppc; ++p )
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                for ( int k = 0; k < 3; ++k, ++particle_counter )
                {
                    positions( particle_counter, 0 ) =
                        grid_min[0] + grid_delta[0] * ( 0.5 + i );
                    positions( particle_counter, 1 ) =
                        grid_min[1] + grid_delta[1] * ( 0.5 + j );
                    positions( particle_counter, 2 ) =
                        grid_min[2] + grid_delta[2] * ( 0.5 + k );
                }


}



int main( int argc, char* argv[] )
{

    Kokkos::ScopeGuard scope_guard( argc, argv );

    freely_moving_rigid_body();

    return 0;

}
