#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <set>
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
int build_element_adjacency_structure(Mesh_t& mesh, std::vector<idx_t>& elemPtr, std::vector<idx_t>& adjacencyList, int& num_elems_per_rank, int elem_start_id, int processRank) {
    
    // Initialize elemPtr with the correct size 
    elemPtr.resize(num_elems_per_rank + 1);


    // Initialize all values in elemPtr to 0
    for (int i = 0; i < elemPtr.size(); i++) {
        elemPtr[i] = 0;
    }

    // Resize adjacencyList with the correct size
    adjacencyList.resize(mesh.num_nodes_in_elem * num_elems_per_rank);

    // Initialize all values in adjacencyList to 0
    for (int i = 0; i < adjacencyList.size(); i++) {

        adjacencyList[i] = 0;
    }
    
    // Fill elemPtr with the correct values
    for (int i = 0; i < num_elems_per_rank; i++) {
        elemPtr[i+1] = elemPtr[i] + mesh.num_nodes_in_elem;
    }

    // Build the adjacency structure
    for (int i = 0; i < num_elems_per_rank; i++) {
        for (int j = 0; j < mesh.num_nodes_in_elem; j++) {
            adjacencyList[elemPtr[i] + j] = mesh.nodes_in_elem.host(elem_start_id + i, j);
        }
    }
    // std::cout << std::endl;
    // std::cout << std::endl;
    // // Print the elemPtr
    // std::cout << "ElemPtr to rank:" << processRank << std::endl;
    // for (int i = 0; i < elemPtr.size(); i++) {
    //     std::cout << elemPtr[i] << " ";
    // }
    // std::cout << std::endl;

    // // Print the adjacency structure
    // std::cout << "Adjacency structure:" << std::endl;
    // for (int i = 0; i < num_elems_per_rank; i++) {
    //     for (int j = 0; j < mesh.num_nodes_in_elem; j++) {
    //         std::cout << adjacencyList[elemPtr[i] + j] << " ";
    //     }
    // }

    return elem_start_id + num_elems_per_rank;
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
    
    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {

        // Get MPI process info
        int processRank, numProcesses;
        MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
        MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

        MPI_Comm comm;
        MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        
        
        Mesh_t initial_mesh;

        // Variables to store mesh information
        Mesh_t mesh;
        node_t node;
        
        size_t total_num_nodes = 0;
        size_t total_num_elems = 0;
        
        size_t local_num_elems = 0;
        size_t local_num_nodes = 0;


        // Initialize the elemDistribution vector
        std::vector<idx_t> elemDistribution(numProcesses+1);

        std::vector<idx_t> initial_elem_local_to_global;

        // Initialize the adjacency structure
        std::vector<idx_t> eind;
        std::vector<idx_t> eptr;

        // Only build mesh on rank 0
        if (processRank == 0) {
            
            // Mesh_t initial_mesh;

            std::vector<double> origin(3);
            origin[0] = 0.0;
            origin[1] = 0.0;
            origin[2] = 0.0;

            std::vector<double> length(3);
            length[0] = 2.0;
            length[1] = 1.0;
            length[2] = 1.0;

            std::vector<int> num_elems_vec(3);
            num_elems_vec[0] = 3;  
            num_elems_vec[1] = 3;
            num_elems_vec[2] = 3;

            build_3d_box(initial_mesh, node, origin, length, num_elems_vec);

            total_num_elems = initial_mesh.num_nodes;
            total_num_elems = initial_mesh.num_elems;
            
            std::cout << "**** Mesh built on rank 0 ****" << std::endl;
            std::cout << "Mesh nodes: " << total_num_nodes << std::endl;
            std::cout << "Mesh elems: " << total_num_elems << std::endl;

            // Communicate mesh information to all ranks
            for (int rank = 1; rank < numProcesses; rank++) {
                MPI_Send(&total_num_nodes, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
                MPI_Send(&total_num_elems, 1, MPI_INT, rank, 1, MPI_COMM_WORLD);
            }

            // Create vector to store the number of elements per rank
            std::vector<idx_t> num_elems_per_rank(numProcesses);

            // Calculate the number of elements per rank, accounting for remainders
            idx_t base_elems_per_proc = total_num_elems / numProcesses;
            idx_t remainder = total_num_elems % numProcesses;
            
            for (int i = 0; i < numProcesses; i++) {
                num_elems_per_rank[i] = base_elems_per_proc;
                if (i < remainder) {
                    num_elems_per_rank[i]++;
                }
            }
            elemDistribution[0] = 0;
            for (int i = 1; i < numProcesses + 1; i++) {
                elemDistribution[i] = elemDistribution[i-1] + num_elems_per_rank[i-1];
            }
            // Print the elemDistribution
            std::cout << "ElemDistribution: ";
            for (int i = 0; i < numProcesses + 1; i++) {
                std::cout << elemDistribution[i] << " ";
            }
            std::cout << std::endl;


            local_num_elems = num_elems_per_rank[0];

            for(int i = 1; i < numProcesses; i++){
                MPI_Send(&num_elems_per_rank[i], 1, MPI_INT, i, 2, MPI_COMM_WORLD);
            }

            // Print the number of elements per rank
            std::cout << "Number of elements per rank: ";
            for (int i = 0; i < numProcesses; i++) {
                std::cout << num_elems_per_rank[i] << " ";
            }
            std::cout << std::endl;


            // Initialize the initial_elem_local_to_global vector
            initial_elem_local_to_global.resize(num_elems_per_rank[0]);
            for (int i = 0; i < num_elems_per_rank[0]; i++) {
                initial_elem_local_to_global[i] = i;
            }

            // Build the adjacency structure for rank 0
            int next_elem_start_id = build_element_adjacency_structure(initial_mesh, eptr, eind, num_elems_per_rank[0], 0, 0);


            // Build the adjacency structure for the other ranks
            // Initialize the adjacency structure
            std::vector<idx_t> local_eind;
            std::vector<idx_t> local_eptr;

            std::vector<idx_t> tmp_elem_local_to_global;

            for (int i = 1; i < numProcesses; i++) {
                next_elem_start_id = build_element_adjacency_structure(initial_mesh, local_eptr, local_eind, num_elems_per_rank[i], next_elem_start_id, i);

                tmp_elem_local_to_global.resize(num_elems_per_rank[i]);
                for (int j = 0; j < num_elems_per_rank[i]; j++) {
                    tmp_elem_local_to_global[j] = next_elem_start_id + j;
                }

                // First send sizes
                int eptr_size = local_eptr.size();
                int eind_size = local_eind.size();
                MPI_Send(&eptr_size, 1, MPI_INT, i, 3, MPI_COMM_WORLD);
                MPI_Send(&eind_size, 1, MPI_INT, i, 4, MPI_COMM_WORLD);

                // Then send actual data
                MPI_Send(local_eptr.data(), eptr_size, MPI_INT, i, 5, MPI_COMM_WORLD);
                MPI_Send(local_eind.data(), eind_size, MPI_INT, i, 6, MPI_COMM_WORLD);

                MPI_Send(tmp_elem_local_to_global.data(), num_elems_per_rank[i], MPI_INT, i, 7, MPI_COMM_WORLD);
            }
        }

        // Other ranks need to receive the information
        else {
            MPI_Recv(&total_num_nodes, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&total_num_elems, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Recv(&local_num_elems, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            eptr.resize(local_num_elems + 1);
            eind.resize(8 * local_num_elems); // WARNING: Assumed 8 nodes per element, FIX LATER

            // First receive sizes
            int eptr_size, eind_size;
            MPI_Recv(&eptr_size, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&eind_size, 1, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Resize vectors accordingly
            eptr.resize(eptr_size);
            eind.resize(eind_size);

            // Receive the actual data
            MPI_Recv(eptr.data(), eptr_size, MPI_INT, 0, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(eind.data(), eind_size, MPI_INT, 0, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            initial_elem_local_to_global.resize(local_num_elems);
            MPI_Recv(initial_elem_local_to_global.data(), local_num_elems, MPI_INT, 0, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Broadcast the elemDistribution to all ranks
        MPI_Bcast(elemDistribution.data(), numProcesses+1, MPI_INT, 0, MPI_COMM_WORLD);
        
        MPI_Barrier(MPI_COMM_WORLD);

        if(processRank == 2){
            // Print the elemDistribution
            std::cout << "ElemDistribution: ";
            for (int i = 0; i < numProcesses + 1; i++) {
                std::cout << elemDistribution[i] << " ";
            }
            std::cout << std::endl;
        }   
        
        std::cout << "local_num_elems: " << local_num_elems << std::endl;    
       
        // Set up ParMETIS parameters
        idx_t wgtflag = 0;         // No weights
        idx_t numflag = 0;         // C-style numbering
        idx_t ncon = 1;            // Number of weights per vertex
        idx_t ncommonnodes = 1;    // Number of common nodes needed for adjacency(NOTE: This may need to be 1 for corners, or 2 for edges)
        idx_t nparts = numProcesses;
        std::vector<real_t> tpwgts(nparts, 1.0/nparts);  // Target partition weights
        std::vector<real_t> ubvec(ncon, 1.05);           // Load imbalance tolerance
        std::vector<idx_t> options(3, 0);                 // Options array
        idx_t edgecut;                                    // Output: Number of edges cut
        std::vector<idx_t> part(local_num_elems);        // Output: Partition vector


        int ret = ParMETIS_V3_PartMeshKway(
            elemDistribution.data(),  // elmdist: element distribution array
            eptr.data(),             // eptr: element pointer array
            eind.data(),             // eind: element connectivity array
            nullptr,                 // elmwgt: element weights (nullptr for no weights)
            &wgtflag,                // wgtflag: weight flag
            &numflag,                // numflag: numbering flag
            &ncon,                   // ncon: number of constraints
            &ncommonnodes,           // ncommonnodes: nodes needed for adjacency
            &nparts,                 // nparts: number of partitions
            tpwgts.data(),          // tpwgts: target partition weights
            ubvec.data(),           // ubvec: load imbalance tolerances
            options.data(),         // options: options array
            &edgecut,               // edgecut: output edge cut
            part.data(),            // part: output partition array
            &comm                   // comm: MPI communicator
        );
 
        // std::cout << "Rank " << processRank << " part.size(): " << part.size() << std::endl;

        // Check the result
        if (ret != METIS_OK) {
            std::cout << "Rank " << processRank << ": ParMETIS returned error: " << ret << std::endl;
            MPI_Abort(comm, 1);
        } 
            
        // Count elements assigned to each partition
        std::vector<idx_t> local_partition_counts(numProcesses, 0);
        for (idx_t i = 0; i < local_num_elems; i++) {
            local_partition_counts[part[i]]++;
        }

        // Gather all counts to rank 0
        std::vector<idx_t> global_partition_counts(numProcesses, 0);
        MPI_Reduce(local_partition_counts.data(), global_partition_counts.data(), 
                    numProcesses, IDX_T, MPI_SUM, 0, comm);

        // Print results on rank 0
        if (processRank == 0) {
            std::cout << "\nParMETIS partitioning results:" << std::endl;
            std::cout << "Edge cut: " << edgecut << std::endl;
            std::cout << "Element distribution:" << std::endl;
            for (int i = 0; i < numProcesses; i++) {
                std::cout << "Partition " << i << ": " << 
                    global_partition_counts[i] << " elements" << std::endl;
            }
        }

        for(int i = 0; i < numProcesses; i++){

            if(processRank == i){

                //Pinrt part for this rank
                std::cout << "Rank " << processRank << " part: ";
                for (int i = 0; i < local_num_elems; i++) {
                    std::cout << part[i] << " ";
                }
                std::cout << std::endl;
                

                // Printing elem_local_to_global for this rank
                std::cout << "Rank " << processRank << " initial_elem_local_to_global: ";
                for (int i = 0; i < local_num_elems; i++) {
                    std::cout << initial_elem_local_to_global[i] << " ";
                }
                std::cout << std::endl;
            }
        }
        


    } // end Kokkos initialize

    
    // Finalize Kokkos
    Kokkos::finalize();
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}