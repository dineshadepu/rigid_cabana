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

using DataTypes = Cabana::MemberTypes<double[3], // position(0)
                                      int, // ids(1)
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

// auto aosoa_ids = Cabana::slice<1>          ( aosoa,    "aosoa_ids");
// auto aosoa_position = Cabana::slice<0>     ( aosoa,    "aosoa_position");
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


using ListAlgorithm = Cabana::FullNeighborTag;
using ListType =
  Cabana::VerletList<MemorySpace, ListAlgorithm, Cabana::VerletLayout2D>;

void set_fluid_properties_of_aosoa(AoSoAType aosoa, std::vector<double>& xf, std::vector<double>& yf, std::vector<double>& zf, std::vector<int>& limits)
{
  /*
    Get the particle ids, coordinates, velocity, radius
  */
  auto positions = Cabana::slice<0>( aosoa, "positions" );
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
  std::vector<int> is_fluid = { 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> fluid_limits = {0, 4};

  // create boundary particle positions
  std::vector<double> xb = {0., 1., 2., 3.};
  std::vector<double> yb = {-1., -1., -1., -1.};
  std::vector<double> zb = {0., 0., 0., 0.};
  std::vector<int> is_boundary = { 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> boundary_limits = {4, 8};

  // create rigid body (bodies) particle positions
  std::vector<double> x_rb = {3., 4., 3., 4., 3., 4., 3., 4.};
  std::vector<double> y_rb = {3., 3., 4., 4., 7., 7., 8., 8.};
  std::vector<double> z_rb = {0., 0., 0., 0., 0., 0., 0., 0.};
  std::vector<int> is_rb = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> body_id = { -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1};
  std::vector<double> m_array = { -1, -1, -1, -1, -1, -1, -1, -1, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1};
  std::vector<int> rb_limits = {8, 16};

  // sum all the number of particles and create aosoa
  AoSoAType aosoa( "particles", num_particles );

  auto ids = Cabana::slice<1>( aosoa, "ids" );
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
  auto aosoa_is_fluid = Cabana::slice<8>    ( aosoa,    "aosoa_is_fluid");
  auto aosoa_is_boundary = Cabana::slice<9>    ( aosoa,    "aosoa_is_boundary");
  auto aosoa_is_rb = Cabana::slice<10>       ( aosoa,    "aosoa_is_rb");
  auto body_ids = Cabana::slice<12>( aosoa, "body_id" );
  // initialize the rigid body properties
  for ( std::size_t i = 0; i < num_particles; ++i )
    {
      aosoa_is_fluid ( i ) = is_fluid[i];
      aosoa_is_boundary ( i ) = is_boundary[i];
      aosoa_is_rb ( i ) = is_rb[i];
      body_ids(i) = body_id[i];
    }
  return aosoa;
}


void setup_rigid_body_properties(auto aosoa, auto rb, auto index_limits){
  // auto aosoa_ids = Cabana::slice<0>          ( aosoa,    "aosoa_ids");
  auto aosoa_position = Cabana::slice<0>     ( aosoa,    "aosoa_position");
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
  // auto com_total_mass_func = KOKKOS_LAMBDA( const int i, double& total_m, double& cm_x, double& cm_y, double& cm_z)
  //   {
  //     auto m_i = aosoa_mass( i );
  //     total_m += m_i;
  //     cm_x += m_i * aosoa_position(i, 0);
  //     cm_y += m_i * aosoa_position(i, 1);
  //     cm_z += m_i * aosoa_position(i, 2);
  //   };

  // for ( std::size_t i = 0; i < rb_position.size(); ++i )
  //   {
  //     double total_mass = 0.0;
  //     double cm_x = 0.0;
  //     double cm_y = 0.0;
  //     double cm_z = 0.0;

  //     std::cout << "rb limits are: " << rb_limits(i, 0) << " " << rb_limits(i, 1) << "\n";

  //     Kokkos::RangePolicy<ExecutionSpace> policy( rb_limits(i, 0), rb_limits(i, 1) );
  //     Kokkos::parallel_reduce(
  //                             "COM_and_total_mass_computation", policy,
  //                             com_total_mass_func, total_mass, cm_x, cm_y, cm_z);
  //     rb_mass(i) = total_mass;
  //     rb_position(i, 0) = cm_x / total_mass;
  //     rb_position(i, 1) = cm_y / total_mass;
  //     rb_position(i, 2) = cm_z / total_mass;
  //   }

  auto parallel_total_mass_func = KOKKOS_LAMBDA( const int i )
    {
      rb_mass(i) = 0.;
      rb_position(i, 0) = 0.;
      rb_position(i, 1) = 0.;
      rb_position(i, 2) = 0.;

      for ( std::size_t j = rb_limits(i, 0); j < rb_limits(i, 1); ++j )
    {
      auto m_j = aosoa_mass( j );
      rb_mass(i) += m_j;
      rb_position(i, 0) += m_j * aosoa_position(j, 0);
      rb_position(i, 1) += m_j * aosoa_position(j, 1);
      rb_position(i, 2) += m_j * aosoa_position(j, 2);
    }
      rb_position(i, 0) /= rb_mass(i);
      rb_position(i, 1) /= rb_mass(i);
      rb_position(i, 2) /= rb_mass(i);

    };
  Kokkos::RangePolicy<ExecutionSpace> policy_tm( 0, rb_mass.size());
  Kokkos::parallel_for( "CabanaRB:RBSetup:TotalMass", policy_tm,
                        parallel_total_mass_func );

  for ( std::size_t i = 0; i < rb_position.size(); ++i )
    {
      std::cout << "\n";
      std::cout << "\n";
      std::cout << "rb mass: " << rb_mass(i) << "\n";
      std::cout << "rb position: " << rb_position(i, 0) << ", " << rb_position(i, 1) << "\n";
      std::cout << "\n";
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

void compute_force_on_rigid_body_particles(auto aosoa, auto dt,
                       ListType * verlet_list_source,
                       auto index_limits){
  auto aosoa_ids = Cabana::slice<1>          ( aosoa,    "aosoa_ids");
  auto aosoa_position = Cabana::slice<0>     ( aosoa,    "aosoa_position");
  auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "aosoa_velocity");
  auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "aosoa_mass");
  auto aosoa_is_rb = Cabana::slice<10>       ( aosoa,    "aosoa_is_rb");
  auto aosoa_radius = Cabana::slice<11>      ( aosoa,    "aosoa_radius");
  auto aosoa_body_id = Cabana::slice<12>     ( aosoa,    "aosoa_body_id");
  auto aosoa_frc_dem = Cabana::slice<14>     ( aosoa,    "aosoa_frc_dem");

  auto dem_contact_force_kernel = KOKKOS_LAMBDA( const int i, const int j )
    {
      if (aosoa_is_rb( j ) == 1 && aosoa_body_id( i ) != aosoa_body_id( i )){
    const double mass_i = aosoa_mass( i );
    const double mass_j = aosoa_mass( j );
    const double radius_i = aosoa_radius( i );
    const double radius_j = aosoa_radius( j );

    const double x_i = aosoa_position( i, 0 );
    const double y_i = aosoa_position( i, 1 );
    const double z_i = aosoa_position( i, 2 );

    const double x_j = aosoa_position( j, 0 );
    const double y_j = aosoa_position( j, 1 );
    const double z_j = aosoa_position( j, 2 );

    const double xij = x_i - x_j;
    const double yij = y_i - y_j;
    const double zij = z_i - z_j;

    const double rsq = xij * xij + yij * yij + zij * zij;
    const double dist = sqrt(rsq);

    const double nij_x = xij / dist;
    const double nij_y = yij / dist;
    const double nij_z = zij / dist;

    const double u_i = aosoa_velocity( i, 0 );
    const double v_i = aosoa_velocity( i, 1 );
    const double w_i = aosoa_velocity( i, 2 );

    const double u_j = aosoa_velocity( j, 0 );
    const double v_j = aosoa_velocity( j, 1 );
    const double w_j = aosoa_velocity( j, 2 );

    // const double wx_i = aosoa_ang_velocity( i, 0 );
    // const double wy_i = aosoa_ang_velocity( i, 1 );
    // const double wz_i = aosoa_ang_velocity( i, 2 );

    // const double wx_j = aosoa_ang_velocity( j, 0 );
    // const double wy_j = aosoa_ang_velocity( j, 1 );
    // const double wz_j = aosoa_ang_velocity( j, 2 );

    // const double u1 = u_i + (nij_y * wz_i - nij_z * wy_i) * radius_i;
    // const double u2 = -u_j + (nij_y * wz_j - nij_z * wy_j) * radius_j;
    // const double uij = u1 + u2;
    // const double v1 = v_i - (nij_x * wz_i - nij_z * wx_i) * radius_i;
    // const double v2 = -v_j - (nij_x * wz_j - nij_z * wx_j) * radius_j;
    // const double vij = v1 + v2;
    // const double w1 = w_i - (nij_x * wy_i - nij_y * wx_i) * radius_i;
    // const double w2 = -w_j - (nij_x * wy_j - nij_y * wx_j) * radius_j;
    // const double wij = w1 + w2;

    // const double vn = uij * nij_x + vij * nij_y + wij * nij_z;
    // const double vn_x = vn * nij_x;
    // const double vn_y = vn * nij_y;
    // const double vn_z = vn * nij_z;

    const double overlap = radius_i + radius_j - dist;

    if (overlap > 0){
      /*
        ############################
        # normal force computation #
        ############################
      */
      // // Compute stiffness
      // // effective Young's modulus
      // double tmp_1 = (1. - nu_i**2.) / E_i;
      // double tmp_2 = (1. - nu_j**2.) / E_j;
      // double E_eff = 1. / (tmp_1 + tmp_2);
      // double tmp_1 = 1. / radius_i;
      // double tmp_2 = 1. / radius_j;
      // double R_eff = 1. / (tmp_1 + tmp_2);
      // // Eq 4 [1];
      // double kn = 4. / 3. * E_eff * R_eff**0.5;
      double kn = 1e4;

      // // compute damping coefficient
      // double tmp_1 = log(cor);
      // double tmp_2 = log(cor)**2. + pi**2.;
      // double alpha_1 = -tmp_1 * (5. / tmp_2)**0.5;
      // double tmp_1 = 1. / mass_i;
      // double tmp_2 = 1. / mass_j;
      // double m_eff = 1. / (tmp_1 + tmp_2);
      // double eta = alpha_1 * (m_eff * kn)**0.5 * overlap**0.25;

      double fn = kn * overlap * sqrt(overlap);
      // double fn_x = fn * nij_x - eta * vn_x;
      // double fn_y = fn * nij_y - eta * vn_y;
      // double fn_z = fn * nij_z - eta * vn_z;
      double fn_x = fn * nij_x;
      double fn_y = fn * nij_y;
      double fn_z = fn * nij_z;

      // aosoa_frc_dem particle_1.fn = fn;
      // particle_1.overlap = overlap;
      aosoa_frc_dem (i, 0) += fn_x;
      aosoa_frc_dem (i, 1) += fn_y;
      aosoa_frc_dem (i, 2) += fn_z;
    }
      }
    };

  Kokkos::RangePolicy<ExecutionSpace> policy(index_limits[0], index_limits[1]);


  Cabana::neighbor_parallel_for( policy,
                 dem_contact_force_kernel,
                 *verlet_list_source,
                 Cabana::FirstNeighborsTag(),
                 Cabana::SerialOpTag(),
                 "dem_contact_force_loop" );
  Kokkos::fence();
}


void compute_effective_force_and_torque_on_rigid_body(auto rb, auto aosoa){
  auto aosoa_position = Cabana::slice<0>     ( aosoa,    "aosoa_position");
  auto aosoa_body_id = Cabana::slice<12>     ( aosoa,    "aosoa_body_id");
  auto aosoa_frc_dem = Cabana::slice<14>     ( aosoa,    "aosoa_frc_dem");

  auto rb_limits = Cabana::slice<1>       ( rb,      "rb_limits");
  auto rb_position = Cabana::slice<2>     ( rb,    "rb_position");
  auto rb_force = Cabana::slice<4>        ( rb,       "rb_force");
  auto rb_torque = Cabana::slice<5>       ( rb,      "rb_torque");
  auto rb_mass = Cabana::slice<11>        ( rb,        "rb_mass");

  auto total_force_torque_func = KOKKOS_LAMBDA( const int i )
    {
      rb_force(i, 0) = 0.;
      rb_force(i, 1) = 0.;
      rb_force(i, 2) = 0.;

      rb_torque(i, 0) = 0.;
      rb_torque(i, 1) = 0.;
      rb_torque(i, 2) = 0.;

      for ( std::size_t j = rb_limits(i, 0); j < rb_limits(i, 1); ++j )
    {
      double fx_j = aosoa_frc_dem(j, 0);
      double fy_j = aosoa_frc_dem(j, 1);
      double fz_j = aosoa_frc_dem(j, 2);

      rb_force(i, 0) += fx_j;
      rb_force(i, 1) += fy_j;
      rb_force(i, 2) += fz_j;

      double dx = aosoa_position( j, 0 ) - rb_position( i, 0 );
      double dy = aosoa_position( j, 1 ) - rb_position( i, 1 );
      double dz = aosoa_position( j, 2 ) - rb_position( i, 2 );

      rb_torque(i, 0) += dy * fz_j - dz * fy_j;
      rb_torque(i, 1) += dz * fx_j - dx * fz_j;
      rb_torque(i, 2) += dx * fy_j - dy * fx_j;
    }
    };

  Kokkos::RangePolicy<ExecutionSpace> policy( 0, rb_mass.size());
  Kokkos::parallel_for( "CabanaRB:RB:TotalForceTorque", policy,
                        total_force_torque_func );
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

  auto aosoa_position = Cabana::slice<0>( aosoa,    "aosoa_position");


  // set the force on the particles of the rigid body
  std::vector<double> frc_dem_x = { 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., -1., -1., 1., 1., 1., 1.};
  std::vector<double> frc_dem_y = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
  std::vector<double> frc_dem_z = { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0., 0., 0.};

  // sum all the number of particles and create aosoa
  // AoSoAType aosoa( "particles", num_particles );

  auto aosoa_frc_dem = Cabana::slice<14>     ( aosoa,    "aosoa_frc_dem");
  for ( std::size_t i = 0; i < aosoa.size(); ++i )
    {
      aosoa_frc_dem( i, 0 ) = frc_dem_x[i];
      aosoa_frc_dem( i, 1 ) = frc_dem_y[i];
      aosoa_frc_dem( i, 2 ) = frc_dem_z[i];
    }

  compute_effective_force_and_torque_on_rigid_body(rb, aosoa);

  auto rb_force = Cabana::slice<4>        ( rb,       "rb_force");
  auto rb_torque = Cabana::slice<5>       ( rb,      "rb_torque");

  for ( std::size_t i = 0; i < rb_force.size(); ++i )
    {
      std::cout << "force on body " << i << " " << rb_force(i, 0) << ", " << rb_force(i, 1) << ", " << rb_force(i, 2) << "\n";
      std::cout << "torque on body " << i << " " << rb_torque(i, 0) << ", " << rb_torque(i, 1) << ", " << rb_torque(i, 2) << "\n";
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
