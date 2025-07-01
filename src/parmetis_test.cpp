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
 * 
 * The decomposition process follows these steps:
 * 1. Initialize MPI and Kokkos
 * 2. Build initial mesh on rank 0
 * 3. Distribute mesh information to all ranks
 * 4. Build adjacency structures for ParMETIS
 * 5. Perform ParMETIS partitioning
 * 6. Redistribute elements based on partitioning
 */
int main(int argc, char *argv[]) {
    // Initialize MPI - required for distributed memory parallelism
    MPI_Init(&argc, &argv);
    
    // Initialize Kokkos - required for shared memory parallelism
    Kokkos::initialize(argc, argv);
    {

        // Get MPI process info - each process needs to know its rank and total number of processes
        int processRank, numProcesses;
        MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
        MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

        // Create a duplicate communicator for ParMETIS to use
        MPI_Comm comm;
        MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        
        // Mesh data structures
        Mesh_t initial_mesh;  // Initial mesh built on rank 0
        node_t initial_node;  // Initial node positions on rank 0
        
        Mesh_t mesh;         // Local mesh for each rank
        node_t node;         // Node data structure
        
        // Mesh size information
        size_t total_num_nodes = 0;  // Total nodes in entire mesh
        size_t total_num_elems = 0;  // Total elements in entire mesh
        size_t local_num_elems = 0;  // Elements assigned to this rank
        size_t local_num_nodes = 0;  // Nodes assigned to this rank

        // ParMETIS data structures
        std::vector<idx_t> elemDistribution(numProcesses+1);  // Distribution of elements across ranks
        std::vector<idx_t> initial_elem_local_to_global;      // Mapping from local to global element IDs
        std::vector<idx_t> eind;  // Element connectivity array for ParMETIS
        std::vector<idx_t> eptr;  // Element pointer array for ParMETIS

        // Step 1: Build initial mesh on rank 0
        if (processRank == 0) {
            // Define mesh parameters
            std::vector<double> origin(3, 0.0);  // Origin point of the mesh
            std::vector<double> length(3);       // Lengths in x,y,z directions
            length[0] = 2.0; 
            length[1] = 1.0; 
            length[2] = 1.0;
            
            // Number of elements in each direction
            std::vector<int> num_elems_vec(3);
            num_elems_vec[0] = 2;  
            num_elems_vec[1] = 2;
            num_elems_vec[2] = 2;

            // Build the initial 3D box mesh
            build_3d_box(initial_mesh, initial_node, origin, length, num_elems_vec);

            // Get total mesh size
            total_num_elems = initial_mesh.num_elems;
            total_num_nodes = initial_mesh.num_nodes;
            
            // Print mesh information
            std::cout << "**** Mesh built on rank 0 ****" << std::endl;
            std::cout << "Mesh nodes: " << total_num_nodes << std::endl;
            std::cout << "Mesh elems: " << total_num_elems << std::endl;

            // Step 2: Distribute mesh information to all ranks
            // Send total mesh size to all other ranks
            for (int rank = 1; rank < numProcesses; rank++) {
                MPI_Send(&total_num_nodes, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
                MPI_Send(&total_num_elems, 1, MPI_INT, rank, 1, MPI_COMM_WORLD);
            }

            // Step 3: Calculate element distribution across ranks
            // Create vector to store the number of elements per rank
            std::vector<idx_t> num_elems_per_rank(numProcesses);

            // Calculate base number of elements per rank and remainder
            idx_t base_elems_per_proc = total_num_elems / numProcesses;
            idx_t remainder = total_num_elems % numProcesses;
            
            // Distribute elements evenly, with remainder distributed to first few ranks
            for (int i = 0; i < numProcesses; i++) {
                num_elems_per_rank[i] = base_elems_per_proc;
                if (i < remainder) {
                    num_elems_per_rank[i]++;
                }
            }

            // Build element distribution array for ParMETIS
            elemDistribution[0] = 0;
            for (int i = 1; i < numProcesses + 1; i++) {
                elemDistribution[i] = elemDistribution[i-1] + num_elems_per_rank[i-1];
            }

            // Print distribution information
            std::cout << "ElemDistribution: ";
            for (int i = 0; i < numProcesses + 1; i++) {
                std::cout << elemDistribution[i] << " ";
            }
            std::cout << std::endl;

            // Set local element count for rank 0
            local_num_elems = num_elems_per_rank[0];

            // Send number of elements to other ranks
            for(int i = 1; i < numProcesses; i++){
                MPI_Send(&num_elems_per_rank[i], 1, MPI_INT, i, 2, MPI_COMM_WORLD);
            }

            // Print element distribution across ranks
            std::cout << "Number of elements per rank: ";
            for (int i = 0; i < numProcesses; i++) {
                std::cout << num_elems_per_rank[i] << " ";
            }
            std::cout << std::endl;

            // Initialize local to global mapping for rank 0
            initial_elem_local_to_global.resize(num_elems_per_rank[0]);
            for (int i = 0; i < num_elems_per_rank[0]; i++) {
                initial_elem_local_to_global[i] = i;
            }

            // Step 4: Build adjacency structures for ParMETIS
            // Build adjacency structure for rank 0
            int next_elem_start_id = build_element_adjacency_structure(initial_mesh, eptr, eind, num_elems_per_rank[0], 0, 0);

            // Build and send adjacency structures for other ranks
            std::vector<idx_t> local_eind;
            std::vector<idx_t> local_eptr;
            std::vector<idx_t> tmp_elem_local_to_global;

            for (int i = 1; i < numProcesses; i++) {
                
                // Build adjacency structure for this rank
                tmp_elem_local_to_global.resize(num_elems_per_rank[i]);
                for (int j = 0; j < num_elems_per_rank[i]; j++) {
                    tmp_elem_local_to_global[j] = next_elem_start_id + j;
                }

                next_elem_start_id = build_element_adjacency_structure(initial_mesh, local_eptr, local_eind, num_elems_per_rank[i], next_elem_start_id, i);


                // Send sizes first
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
        // Step 2 (continued): Other ranks receive mesh information
        else {
            // Receive total mesh size
            MPI_Recv(&total_num_nodes, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&total_num_elems, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Receive number of elements for this rank
            MPI_Recv(&local_num_elems, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Allocate space for adjacency structures
            eptr.resize(local_num_elems + 1);
            eind.resize(8 * local_num_elems); // WARNING: Assumed 8 nodes per element, FIX LATER

            // Receive sizes first
            int eptr_size, eind_size;
            MPI_Recv(&eptr_size, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&eind_size, 1, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Resize vectors
            eptr.resize(eptr_size);
            eind.resize(eind_size);

            // Receive actual data
            MPI_Recv(eptr.data(), eptr_size, MPI_INT, 0, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(eind.data(), eind_size, MPI_INT, 0, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Receive local to global mapping
            initial_elem_local_to_global.resize(local_num_elems);
            MPI_Recv(initial_elem_local_to_global.data(), local_num_elems, MPI_INT, 0, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Broadcast element distribution to all ranks
        MPI_Bcast(elemDistribution.data(), numProcesses+1, MPI_INT, 0, MPI_COMM_WORLD);
        
        MPI_Barrier(MPI_COMM_WORLD);

        // Step 5: Set up ParMETIS parameters
        idx_t wgtflag = 0;         // No weights
        idx_t numflag = 0;         // C-style numbering
        idx_t ncon = 1;            // Number of weights per vertex
        idx_t ncommonnodes = 1;    // Number of common nodes needed for adjacency
        idx_t nparts = numProcesses;
        std::vector<real_t> tpwgts(nparts, 1.0/nparts);  // Target partition weights
        std::vector<real_t> ubvec(ncon, 1.05);           // Load imbalance tolerance
        std::vector<idx_t> options(3, 0);                 // Options array
        idx_t edgecut;                                    // Output: Number of edges cut
        std::vector<idx_t> part(local_num_elems);        // Output: Partition vector

        // Call ParMETIS to perform partitioning
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
 
        // Check ParMETIS result
        if (ret != METIS_OK) {
            std::cout << "Rank " << processRank << ": ParMETIS returned error: " << ret << std::endl;
            MPI_Abort(comm, 1);
        } 
            
        // Step 6: Analyze partitioning results
        // Count elements assigned to each partition locally
        std::vector<idx_t> local_partition_counts(numProcesses, 0);
        for (idx_t i = 0; i < local_num_elems; i++) {
            local_partition_counts[part[i]]++;
        }

        // Gather all counts to rank 0 for analysis
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

        MPI_Barrier(MPI_COMM_WORLD);

        for(int i = 0; i < numProcesses; i++){
            if(processRank == i){

                // //Pinrt part for this rank
                // std::cout << "Rank " << processRank << " part: ";
                // for (int i = 0; i < local_num_elems; i++) {
                //     std::cout << part[i] << " ";
                // }
                // std::cout << std::endl;
                

                // Printing elem_local_to_global for this rank
                std::cout << "Rank " << processRank << " initial_elem_local_to_global: ";
                for (int i = 0; i < local_num_elems; i++) {
                    std::cout << initial_elem_local_to_global[i] << " ";
                }
                std::cout << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // Collect all element IDs that should exist on this rank based on the ParMETIS decomposition
        std::vector<idx_t> my_elements_to_receive; // Will store global IDs of elements assigned to this rank
        std::vector<int> send_counts(numProcesses, 0);  // How many elements to send to each rank
        std::vector<std::vector<idx_t>> elements_to_send(numProcesses); // The actual elements to send to each rank

        // Determine which elements need to be sent where
        for (idx_t i = 0; i < local_num_elems; i++) {
            int target_rank = part[i];  // The rank this element should go to
            send_counts[target_rank]++;
            elements_to_send[target_rank].push_back(initial_elem_local_to_global[i]);
        }

        // Exchange information about how many elements each rank will receive
        std::vector<int> recv_counts(numProcesses, 0);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

        // Calculate displacements for MPI_Alltoallv
        std::vector<int> send_displs(numProcesses, 0);
        std::vector<int> recv_displs(numProcesses, 0);
        for (int i = 1; i < numProcesses; i++) {
            send_displs[i] = send_displs[i-1] + send_counts[i-1];
            recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
        }

        // Calculate the total number of elements to receive
        int total_recv_count = 0;
        for (int i = 0; i < numProcesses; i++) {
            total_recv_count += recv_counts[i];
        }

        // Flatten send data for MPI_Alltoallv
        std::vector<idx_t> all_elements_to_send;
        for (int i = 0; i < numProcesses; i++) {
            all_elements_to_send.insert(all_elements_to_send.end(), elements_to_send[i].begin(), elements_to_send[i].end());
        }

        // Prepare receive buffer
        my_elements_to_receive.resize(total_recv_count);

        // Exchange element IDs using MPI_Alltoallv
        // This collective communication operation performs an all-to-all personalized exchange
        // where each process sends different amounts of data to each other process
        MPI_Alltoallv(
            all_elements_to_send.data(),    // [in] Buffer containing all elements to be sent
            send_counts.data(),             // [in] Array of length numProcesses, send_counts[i] = number of elements to send to rank i
            send_displs.data(),             // [in] Array of length numProcesses, send_displs[i] = displacement in send buffer for data going to rank i
            MPI_INT,                        // [in] MPI datatype for the elements
            my_elements_to_receive.data(),  // [out] Buffer to store received elements
            recv_counts.data(),             // [in] Array of length numProcesses, recv_counts[i] = number of elements to receive from rank i
            recv_displs.data(),             // [in] Array of length numProcesses, recv_displs[i] = displacement in receive buffer for data from rank i
            MPI_INT,                        // [in] MPI datatype for the elements
            comm                            // [in] MPI communicator
        );

        MPI_Barrier(MPI_COMM_WORLD);

        // Sort received elements for easier lookup
        std::sort(my_elements_to_receive.begin(), my_elements_to_receive.end());

        // Print received elements for verification
        std::cout << "Rank " << processRank << " received " << my_elements_to_receive.size() 
                << " elements after redistribution: ";
        for (size_t i = 0; i < std::min(size_t(20), my_elements_to_receive.size()); i++) {
            std::cout << my_elements_to_receive[i] << " ";
        }
        if (my_elements_to_receive.size() > 20) {
            std::cout << "... (and " << my_elements_to_receive.size() - 20 << " more)";
        }
 
        std::cout << std::endl;
        std::cout << std::endl;
        
        MPI_Barrier(MPI_COMM_WORLD);

        // if(processRank == 0){
            
        //     // Build list of nodes that exists on this rank using the parmetis partitioning
        //     std::vector<idx_t> nodes_to_receive;

        //     std::cout<<"Rank 0: local_num_elems: "<<local_num_elems<<std::endl;
        //     std::cout<<"Initial mesh num_nodes_in_elem: "<<initial_mesh.num_nodes_in_elem<<std::endl;

        //     for(int i = 0; i < local_num_elems; i++){

        //         int elem_gid = my_elements_to_receive[i];
        //         std::cout<<"Rank 0: elem_gid: "<<elem_gid<<std::endl;
        //         for(int j = 0; j < 8; j++){
        //             idx_t node_gid = initial_mesh.nodes_in_elem(my_elements_to_receive[i], j);
        //             std::cout<<node_gid<< ", " ;
        //             nodes_to_receive.push_back(node_gid);
        //         }
        //         std::cout<<std::endl;
        //     }

        //     // Remove duplicates from nodes_to_receive
        //     std::sort(nodes_to_receive.begin(), nodes_to_receive.end());
        //     nodes_to_receive.erase(std::unique(nodes_to_receive.begin(), nodes_to_receive.end()), nodes_to_receive.end());

        //     // Print nodes_to_receive
        //     std::cout << "Rank " << processRank << " nodes_to_receive: ";
        //     for(int i = 0; i < nodes_to_receive.size(); i++){   
        //         std::cout << nodes_to_receive[i] << " ";
        //     }
        //     std::cout << std::endl;

        //     std::cout<<"Rank 0: nodes_to_receive.size(): "<<nodes_to_receive.size()<<std::endl;

        //     // Initialize node data structure
        //     node.initialize(1, nodes_to_receive.size(), 3, {node_state::coords});

        //     // Initialize node coordinates
        //     for(int i = 0; i < nodes_to_receive.size(); i++){
        //         node.coords(0, i, 0) = initial_node.coords(0, nodes_to_receive[i], 0);
        //         node.coords(0, i, 1) = initial_node.coords(0, nodes_to_receive[i], 1);
        //         node.coords(0, i, 2) = initial_node.coords(0, nodes_to_receive[i], 2);
        //     }   
        // }

        // Step 7: Distribute nodes to all ranks based on element ownership
        // This section handles the distribution of node data from rank 0 to all ranks
        // based on which elements each rank owns after ParMETIS partitioning

        std::vector<idx_t> nodes_to_receive;

        // Array to store node IDs needed by each rank
        std::vector<std::vector<idx_t>> nodes_per_rank;
        
        // Array to store node coordinates for each rank
        std::vector<std::vector<double>> node_coords_per_rank;
        
        if(processRank == 0){
            // On rank 0, we need to determine which nodes each rank needs based on their elements
            
            // Initialize the array to hold node lists for each rank
            nodes_per_rank.resize(numProcesses);
            
            // Gather all elements from all ranks to rank 0
            std::vector<int> all_recv_counts(numProcesses);
            MPI_Gather(&total_recv_count, 1, MPI_INT, all_recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            // Calculate total elements and displacements for gathering
            int total_elems = 0;
            std::vector<int> all_recv_displs(numProcesses, 0);
            for(int i = 0; i < numProcesses; i++) {
                total_elems += all_recv_counts[i];
                if(i > 0) {
                    all_recv_displs[i] = all_recv_displs[i-1] + all_recv_counts[i-1];
                }
            }
            
            // Gather all elements to rank 0
            std::vector<idx_t> all_elements(total_elems);
            MPI_Gatherv(my_elements_to_receive.data(), total_recv_count, MPI_INT,
                        all_elements.data(), all_recv_counts.data(), all_recv_displs.data(),
                        MPI_INT, 0, MPI_COMM_WORLD);
            
            // Now build node lists for all ranks
            for(int rank = 0; rank < numProcesses; rank++) {
                // Get the starting index for this rank's elements
                int start_idx = all_recv_displs[rank];
                int count = all_recv_counts[rank];
                
                // Process each element assigned to this rank
                for(int i = 0; i < count; i++) {
                    int elem_gid = all_elements[start_idx + i];
                    
                    // Get the nodes for this element
                    for(int j = 0; j < initial_mesh.num_nodes_in_elem; j++) {
                        idx_t node_gid = initial_mesh.nodes_in_elem.host(elem_gid, j);
                        nodes_per_rank[rank].push_back(node_gid);
                    }
                }
                
                // Remove duplicates from the node list
                std::sort(nodes_per_rank[rank].begin(), nodes_per_rank[rank].end());
                nodes_per_rank[rank].erase(
                    std::unique(nodes_per_rank[rank].begin(), nodes_per_rank[rank].end()), 
                    nodes_per_rank[rank].end()
                );

                // Print info about the node list for this rank
                std::cout << "Rank 0: Preparing to send " << nodes_per_rank[rank].size() 
                          << " nodes to rank " << rank << std::endl;
            }
            
            // Now prepare the coordinate data for each node list
            node_coords_per_rank.resize(numProcesses);
            for(int rank = 0; rank < numProcesses; rank++) {
                // For each node in this rank's list, extract its coordinates
                node_coords_per_rank[rank].resize(nodes_per_rank[rank].size() * 3); // 3 coordinates per node
                
                for(int i = 0; i < nodes_per_rank[rank].size(); i++) {
                    idx_t node_gid = nodes_per_rank[rank][i];
                    node_coords_per_rank[rank][i*3 + 0] = initial_node.coords.host(0, node_gid, 0);
                    node_coords_per_rank[rank][i*3 + 1] = initial_node.coords.host(0, node_gid, 1);
                    node_coords_per_rank[rank][i*3 + 2] = initial_node.coords.host(0, node_gid, 2);
                }
            }
            
            // Rank 0 keeps its own data
            nodes_to_receive.resize(nodes_per_rank[0].size());
            nodes_to_receive = nodes_per_rank[0];
            std::vector<double> node_coords = node_coords_per_rank[0];
            
            // Initialize node data structure for rank 0
            node.initialize(1, nodes_to_receive.size(), 3, {node_state::coords});
            
            // Copy coordinates to the node data structure
            for(int i = 0; i < nodes_to_receive.size(); i++) {
                node.coords.host(0, i, 0) = node_coords[i*3 + 0];
                node.coords.host(0, i, 1) = node_coords[i*3 + 1];
                node.coords.host(0, i, 2) = node_coords[i*3 + 2];
            }
            node.coords.update_device();
            
            // Send node data to all other ranks
            for(int rank = 1; rank < numProcesses; rank++) {
                // Send number of nodes first
                int num_nodes = nodes_per_rank[rank].size();
                MPI_Send(&num_nodes, 1, MPI_INT, rank, 10, MPI_COMM_WORLD);
                
                // Send the list of node IDs
                MPI_Send(nodes_per_rank[rank].data(), num_nodes, MPI_INT, rank, 11, MPI_COMM_WORLD);
                
                // Send the node coordinates (3 coordinates per node)
                MPI_Send(node_coords_per_rank[rank].data(), num_nodes * 3, MPI_DOUBLE, rank, 12, MPI_COMM_WORLD);
            }
            
            // Print summary of node distribution
            std::cout << "Rank 0: Finished distributing nodes to all ranks" << std::endl;
        }
        else {
            // Other ranks need to send their element count to rank 0
            MPI_Gather(&total_recv_count, 1, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
            
            // Send elements to rank 0
            MPI_Gatherv(my_elements_to_receive.data(), total_recv_count, MPI_INT,
                        nullptr, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);
            
            // Receive the number of nodes first
            int num_nodes;
            MPI_Recv(&num_nodes, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Allocate space and receive the list of node IDs
            nodes_to_receive.resize(num_nodes);
            MPI_Recv(nodes_to_receive.data(), num_nodes, MPI_INT, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Allocate space and receive the node coordinates
            std::vector<double> node_coords(num_nodes * 3);
            MPI_Recv(node_coords.data(), num_nodes * 3, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Initialize the node data structure
            node.initialize(1, num_nodes, 3, {node_state::coords});
            
            // Copy the coordinates to the node data structure
            for(int i = 0; i < num_nodes; i++) {
                node.coords.host(0, i, 0) = node_coords[i*3 + 0];
                node.coords.host(0, i, 1) = node_coords[i*3 + 1];
                node.coords.host(0, i, 2) = node_coords[i*3 + 2];
            }
            node.coords.update_device();
        }

        MPI_Barrier(MPI_COMM_WORLD);

        std::cout << std::endl;
        std::cout << std::endl;


        if(processRank == 1){
            // Print node coordinates for verification
            std::cout << "Rank " << processRank << " node coordinates:" << std::endl;
            for(int i = 0; i < node.coords.dims(1); i++) {
                std::cout << "Node " << i << ": " << node.coords.host(0, i, 0) << ", " 
                        << node.coords.host(0, i, 1) << ", " 
                        << node.coords.host(0, i, 2) << std::endl;    
            }
        }



        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        
 


    } // end Kokkos initialize
    // Finalize Kokkos
    Kokkos::finalize();
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}