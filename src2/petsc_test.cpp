#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <set>
#include <mpi.h>
// #include <parmetis.h>
#include <Kokkos_Core.hpp>
#include "matar.h"
// #include "distributed_array.h"
#include "mesh.h"


#include <petsc.h>

/*
mpirun -np 2 ./parmetis_test


*/


// Possible node states, used to initialize node_t
enum class node_state
{
    coords,
    velocity,
    mass,
    temp,
    q_flux,
    force,
};

/////////////////////////////////////////////////////////////////////////////
///
/// \struct node_t
///
/// \brief Stores state information associated with a node
///
/////////////////////////////////////////////////////////////////////////////
struct node_t
{
    DCArrayKokkos<double> coords; ///< Nodal coordinates

    // initialization method (num_rk_storage_bins, num_nodes, num_dims, state to allocate)
    void initialize(size_t num_rk, size_t num_nodes, size_t num_dims, std::vector<node_state> node_states)
    {
        for (auto field : node_states){
            switch(field){
                case node_state::coords:
                    if (coords.size() == 0) this->coords = DCArrayKokkos<double>(num_rk, num_nodes, num_dims, "node_coordinates");
                    break;
                default:
                    std::cout<<"Desired node state not understood in node_t initialize"<<std::endl;
            }
        }
    }; // end method
}; // end node_t

inline int get_id(int i, int j, int k, int num_i, int num_j)
{
    return i + j * num_i + k * num_i * num_j;
}


/////////////////////////////////////////////////////////////////////////////
///
/// \fn build_3d_box
///
/// \brief Builds an unstructured 3D rectilinear mesh
///
/// \param Simulation mesh that is built
/// \param Element state data
/// \param Node state data
/// \param Corner state data
/// \param Simulation parameters
///
/////////////////////////////////////////////////////////////////////////////
void build_3d_box(Mesh_t& mesh,
        node_t&   node,
        std::vector<double> origin,
        std::vector<double> length,
        std::vector<int> num_elems)
    {
        printf(" Creating a 3D box mesh \n");

        const int num_dim = 3;

        // SimulationParamaters.mesh_input.length.update_host();
        const double lx = length[0];
        const double ly = length[1];
        const double lz = length[2];

        // SimulationParamaters.mesh_input.num_elems.update_host();
        const int num_elems_i = num_elems[0];
        const int num_elems_j = num_elems[1];
        const int num_elems_k = num_elems[2];

        const int num_points_i = num_elems_i + 1; // num points in x
        const int num_points_j = num_elems_j + 1; // num points in y
        const int num_points_k = num_elems_k + 1; // num points in y

        const int num_nodes = num_points_i * num_points_j * num_points_k;

        const double dx = lx / ((double)num_elems_i);  // len/(num_elems_i)
        const double dy = ly / ((double)num_elems_j);  // len/(num_elems_j)
        const double dz = lz / ((double)num_elems_k);  // len/(num_elems_k)

        const int total_num_elems = num_elems_i * num_elems_j * num_elems_k;

        int rk_num_bins = 2;

        // initialize mesh node variables
        mesh.initialize_nodes(num_nodes);

         // initialize node state variables, for now, we just need coordinates, the rest will be initialize by the respective solvers
        std::vector<node_state> required_node_state = { node_state::coords };
        node.initialize(rk_num_bins, num_nodes, num_dim, required_node_state);

        // --- Build nodes ---

        // populate the point data structures
        for (int k = 0; k < num_points_k; k++) {
            for (int j = 0; j < num_points_j; j++) {
                for (int i = 0; i < num_points_i; i++) {
                    // global id for the point
                    int node_gid = get_id(i, j, k, num_points_i, num_points_j);

                    // store the point coordinates
                    node.coords.host(0, node_gid, 0) = origin[0] + (double)i * dx;
                    node.coords.host(0, node_gid, 1) = origin[1] + (double)j * dy;
                    node.coords.host(0, node_gid, 2) = origin[2] + (double)k * dz;
                } // end for i
            } // end for j
        } // end for k

        for (int rk_level = 1; rk_level < rk_num_bins; rk_level++) {
            for (int node_gid = 0; node_gid < num_nodes; node_gid++) {
                node.coords.host(rk_level, node_gid, 0) = node.coords.host(0, node_gid, 0);
                node.coords.host(rk_level, node_gid, 1) = node.coords.host(0, node_gid, 1);
                node.coords.host(rk_level, node_gid, 2) = node.coords.host(0, node_gid, 2);
            }
        }
        node.coords.update_device();

        // initialize elem variables
        mesh.initialize_elems(total_num_elems, num_dim);

        // --- Build elems  ---

        // populate the elem center data structures
        for (int k = 0; k < num_elems_k; k++) {
            for (int j = 0; j < num_elems_j; j++) {
                for (int i = 0; i < num_elems_i; i++) {
                    // global id for the elem
                    int elem_gid = get_id(i, j, k, num_elems_i, num_elems_j);

                    // store the point IDs for this elem where the range is
                    // (i:i+1, j:j+1, k:k+1) for a linear hexahedron
                    int this_point = 0;
                    for (int kcount = k; kcount <= k + 1; kcount++) {
                        for (int jcount = j; jcount <= j + 1; jcount++) {
                            for (int icount = i; icount <= i + 1; icount++) {
                                // global id for the points
                                int node_gid = get_id(icount, jcount, kcount,
                                                  num_points_i, num_points_j);

                                // convert this_point index to the FE index convention
                                int this_index = this_point; //convert_point_number_in_Hex(this_point);

                                // store the points in this elem according the the finite
                                // element numbering convention
                                mesh.nodes_in_elem.host(elem_gid, this_index) = node_gid;

                                // increment the point counting index
                                this_point = this_point + 1;
                            } // end for icount
                        } // end for jcount
                    }  // end for kcount
                } // end for i
            } // end for j
        } // end for k

        // update device side
        mesh.nodes_in_elem.update_device();

        // initialize corner variables
        int num_corners = total_num_elems * mesh.num_nodes_in_elem;
        mesh.initialize_corners(num_corners);

        // Build connectivity
        mesh.build_connectivity();
    } // end build_3d_box





int main(int argc, char *argv[]) {
    // Initialize MPI - required for distributed memory parallelism
    MPI_Init(&argc, &argv);
    
    // Initialize Kokkos - required for shared memory parallelism
    Kokkos::initialize(argc, argv);
    {

        // Initialize PETSc
        PetscErrorCode ierr;
        ierr = PetscInitialize(&argc, &argv, nullptr, nullptr); CHKERRQ(ierr);
        
        // Get MPI info
        PetscMPIInt rank, size;
        ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
        ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size); CHKERRQ(ierr);
        
        // Print hello from each process
        ierr = PetscPrintf(PETSC_COMM_WORLD, 
                        "Hello from rank %d of %d processes\n", 
                        rank, size); CHKERRQ(ierr);
        
        // Create a vector
        Vec x;
        PetscInt n = 10;
        ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
        ierr = VecSetSizes(x, PETSC_DECIDE, n); CHKERRQ(ierr);
        ierr = VecSetFromOptions(x); CHKERRQ(ierr);
        
        // Set vector values
        ierr = VecSet(x, 1.0); CHKERRQ(ierr);
        
        // Compute vector norm
        PetscReal norm;
        ierr = VecNorm(x, NORM_2, &norm); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, 
                        "Vector norm: %g\n", norm); CHKERRQ(ierr);
        
        // Clean up
        ierr = VecDestroy(&x); CHKERRQ(ierr);
        
        // Finalize PETSc
        ierr = PetscFinalize(); CHKERRQ(ierr);

    } // end Kokkos initialize
    // Finalize Kokkos
    Kokkos::finalize();
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}