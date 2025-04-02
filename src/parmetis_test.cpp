#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <mpi.h>
#include <parmetis.h>
#include <Kokkos_Core.hpp>
#include "matar.h"
#include "distributed_array.h"
#include "mesh.h"


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


// Function to build adjacency structure required by ParMETIS
void build_adjacency_structure(Mesh_t& mesh, std::vector<idx_t>& adjacencyPointers, std::vector<idx_t>& adjacencyList) {
    // Initialize adjacencyPointers with the correct size (num_nodes + 1)
    adjacencyPointers.resize(mesh.num_nodes + 1);
    adjacencyPointers[0] = 0;

    // First pass: Set up adjacencyPointers using num_nodes_in_node
    size_t totalAdjacencies = 0;
    for (size_t i = 0; i < mesh.num_nodes; ++i) {
        size_t numAdjacent = mesh.num_nodes_in_node(i);
        adjacencyPointers[i + 1] = adjacencyPointers[i] + numAdjacent;
        totalAdjacencies += numAdjacent;
    }

    // Resize adjacencyList to hold all adjacencies
    adjacencyList.resize(totalAdjacencies);

    // Second pass: Fill adjacencyList using nodes_in_node
    size_t currentIndex = 0;
    for (size_t i = 0; i < mesh.num_nodes; ++i) {
        size_t numAdjacent = mesh.num_nodes_in_node(i);
        
        // Copy adjacent nodes
        for (size_t j = 0; j < numAdjacent; ++j) {
            adjacencyList[currentIndex++] = mesh.nodes_in_node(i, j);
        }
    }

    // Debug output
    std::cout << "Adjacency structure built using mesh connectivity:" << std::endl;
    std::cout << "Number of nodes: " << mesh.num_nodes << std::endl;
    std::cout << "Total adjacencies: " << totalAdjacencies << std::endl;
    std::cout << "Average connectivity: " << (double)totalAdjacencies / mesh.num_nodes << std::endl;
}

// Add this function before main()
void validate_parmetis_input(const std::vector<idx_t>& vertexDistribution,
                           const std::vector<idx_t>& adjacencyPointers,
                           const std::vector<idx_t>& adjacencyList,
                           int processRank) {
    if (processRank == 0) {
        std::cout << "\nValidating ParMETIS input:" << std::endl;
        
        // Check vertex distribution is monotonic
        bool is_monotonic = true;
        for (size_t i = 1; i < vertexDistribution.size(); ++i) {
            if (vertexDistribution[i] < vertexDistribution[i-1]) {
                is_monotonic = false;
                std::cout << "Error: vertexDistribution is not monotonically increasing" << std::endl;
                break;
            }
        }
        std::cout << "Vertex distribution monotonic: " << (is_monotonic ? "yes" : "no") << std::endl;

        // Check adjacency pointers are consistent
        bool pointers_valid = true;
        for (size_t i = 0; i < adjacencyPointers.size() - 1; ++i) {
            if (adjacencyPointers[i] > adjacencyPointers[i+1]) {
                pointers_valid = false;
                std::cout << "Error: adjacencyPointers not monotonically increasing at index " << i << std::endl;
                break;
            }
        }
        std::cout << "Adjacency pointers valid: " << (pointers_valid ? "yes" : "no") << std::endl;

        // Check for self-loops and print first few adjacencies
        std::cout << "First few adjacencies:" << std::endl;
        
        for (size_t i = 0; i < std::min(size_t(5), (size_t)vertexDistribution[1]); ++i) {
            std::cout << "Node " << i << " connected to: ";
            for (idx_t j = adjacencyPointers[i]; j < adjacencyPointers[i+1]; ++j) {
                if (adjacencyList[j] == i) {
                    std::cout << "(self-loop!) ";
                }
                std::cout << adjacencyList[j] << " ";
            }
            std::cout << std::endl;
        }
    }
}

/**
 * Main function to demonstrate ParMETIS graph partitioning with MATAR
 */
int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    // Get MPI process info
    int processRank, numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    
    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        // Variables to store mesh information
        Mesh_t mesh;
        node_t node;
        size_t num_nodes = 0;
        size_t num_elems = 0;

        // Only build mesh on rank 0
        if (processRank == 0) {

            std::vector<double> origin(3);
            origin[0] = 0.0;
            origin[1] = 0.0;
            origin[2] = 0.0;

            std::vector<double> length(3);
            length[0] = 2.0;
            length[1] = 1.0;
            length[2] = 1.0;

            std::vector<int> num_elems_vec(3);
            num_elems_vec[0] = 20;  
            num_elems_vec[1] = 20;
            num_elems_vec[2] = 20;

            build_3d_box(mesh, node, origin, length, num_elems_vec);

            num_nodes = mesh.num_nodes;
            num_elems = mesh.num_elems;

            std::cout << "**** Mesh built on rank 0 ****" << std::endl;
            std::cout << "Mesh nodes: " << num_nodes << std::endl;
            std::cout << "Mesh elems: " << num_elems << std::endl;

            std::cout << "Number of processes: " << numProcesses << std::endl;

            // Use METIS to partition the mesh
            // int METIS PartMeshNodal( idx t *ne, idx t *nn, idx t *eptr, idx t *eind, idx t *vwgt, idx t *vsize,
            // idx t *nparts, real t *tpwgts, idx t *options, idx t *objval, idx t *epart, idx t *npart)
            idx_t ne = num_elems;
            idx_t nn = num_nodes;
            std::vector<idx_t> eptr(num_elems + 1);
            std::vector<idx_t> eind(num_elems * mesh.num_nodes_in_elem);
            std::vector<real_t> tpwgts(numProcesses, 1.0/numProcesses);
            std::vector<idx_t> options(METIS_NOPTIONS);
            std::vector<idx_t> epart(num_elems);
            std::vector<idx_t> npart(num_nodes);

            // Initialize options
            METIS_SetDefaultOptions(options.data());
            options[METIS_OPTION_NUMBERING] = 0;
            options[METIS_OPTION_PTYPE] = METIS_PTYPE_RB;
            options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
            options[METIS_OPTION_CONTIG] = 1;

            // Initialize eptr
            eptr[0] = 0;
            for (idx_t i = 0; i < num_elems; i++) {
                eptr[i + 1] = eptr[i] + mesh.num_nodes_in_elem;
            }

            // Fill eind array
            idx_t idx = 0;
            for (idx_t elem_id = 0; elem_id < num_elems; elem_id++) {
                for (idx_t node_lid = 0; node_lid < mesh.num_nodes_in_elem; node_lid++) {
                    eind[idx++] = mesh.nodes_in_elem.host(elem_id, node_lid);
                }
            }

            // Debug print before calling METIS
            std::cout << "Mesh information before METIS call:" << std::endl;
            std::cout << "Number of elements: " << ne << std::endl;
            std::cout << "Number of nodes: " << nn << std::endl;
            std::cout << "Nodes per element: " << mesh.num_nodes_in_elem << std::endl;
            // std::cout << "eptr array: ";
            // for (const auto& v : eptr) std::cout << v << " ";
            // std::cout << std::endl;
            // std::cout << "eind array: ";
            // for (const auto& v : eind) std::cout << v << " ";
            // std::cout << std::endl;

            // Set up options properly
            METIS_SetDefaultOptions(options.data());
            options[METIS_OPTION_NUMBERING] = 0;      // C-style numbering
            options[METIS_OPTION_PTYPE] = METIS_PTYPE_RB;  // Use recursive bisection
            options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
            
            // Call METIS
            idx_t objval;
            idx_t nparts = numProcesses;
            idx_t ret = METIS_PartMeshNodal(
                &ne,                // number of elements
                &nn,               // number of nodes
                eptr.data(),       // element ptr array
                eind.data(),       // element connectivity array
                nullptr,           // vertex weights (no weights)
                nullptr,           // vertex sizes (no sizes)
                &nparts,           // number of parts
                nullptr,           // target weights (nullptr for uniform)
                options.data(),    // options array
                &objval,           // output: objective value
                epart.data(),      // output: element partition vector
                npart.data()       // output: node partition vector
            );

            if (ret != METIS_OK) {
                std::cout << "METIS error code: " << ret << std::endl;
            }

            // // Print the partitioning
            // std::cout << "Partitioning:" << std::endl;
            // for (idx_t i = 0; i < num_nodes; i++) {
            //     std::cout << "Node " << i << " is in partition " << npart[i] << std::endl;
            // }

            // // Print the element partitioning
            // std::cout << "Element partitioning:" << std::endl;
            // for (idx_t i = 0; i < num_elems; i++) {
            //     std::cout << "Element " << i << " is in partition " << epart[i] << std::endl;
            // }
            
            // Count the number of nodes in each partition
            std::vector<idx_t> partition_sizes(numProcesses, 0);
            for (idx_t i = 0; i < num_nodes; i++) {
                partition_sizes[npart[i]]++;
            }   
            
            // Print the number of nodes in each partition
            std::cout << "Number of nodes in each partition:" << std::endl;
            for (idx_t i = 0; i < numProcesses; i++) {
                std::cout << "Partition " << i << ": " << partition_sizes[i] << " nodes" << std::endl;
            }
            
            // Count the number of elements in each partition
            std::vector<idx_t> element_sizes(numProcesses, 0);
            for (idx_t i = 0; i < num_elems; i++) {
                element_sizes[epart[i]]++;
            }   
            
            // Print the number of elements in each partition
            std::cout << "Number of elements in each partition:" << std::endl;
            for (idx_t i = 0; i < numProcesses; i++) {
                std::cout << "Partition " << i << ": " << element_sizes[i] << " elements" << std::endl;
            }   


        }

        
    }
    
    // Finalize Kokkos
    Kokkos::finalize();
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}
