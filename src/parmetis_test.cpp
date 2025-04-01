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
        length[0] = 1.0;
        length[1] = 1.0;
        length[2] = 1.0;

            std::vector<int> num_elems_vec(3);
            num_elems_vec[0] = 3;  
            num_elems_vec[1] = 3;
            num_elems_vec[2] = 3;

        build_3d_box(mesh, 
                     node, 
                     origin,
                     length,
                        num_elems_vec);

            num_nodes = mesh.num_nodes;
            num_elems = mesh.num_elems;

            std::cout << "**** Mesh built on rank 0 ****" << std::endl;
            std::cout << "Mesh nodes: " << num_nodes << std::endl;
            std::cout << "Mesh elems: " << num_elems << std::endl;
        }

        // Broadcast mesh size information to all ranks
        MPI_Bcast(&num_nodes, 1, MPI_UNSIGNED_LONG, 0, comm);
        MPI_Bcast(&num_elems, 1, MPI_UNSIGNED_LONG, 0, comm);

        // Build adjacency structure for the mesh (only on rank 0)
        // adjacencyPointers: Array of size num_nodes + 1 that stores the starting index in adjacencyList for each vertex
        // adjacencyPointers[i] points to the start of vertex i's adjacency list in adjacencyList
        // adjacencyPointers[num_nodes] contains the total size of adjacencyList
        std::vector<idx_t> adjacencyPointers;
        
        // adjacencyList: Array that stores the actual adjacency information
        // For each vertex i, adjacencyList[adjacencyPointers[i] to adjacencyPointers[i+1]-1] contains
        // the indices of vertices that are adjacent to vertex i
        std::vector<idx_t> adjacencyList;
        
        if (processRank == 0) {
            adjacencyPointers.resize(num_nodes + 1);
        adjacencyPointers[0] = 0;
            build_adjacency_structure(mesh, adjacencyPointers, adjacencyList);
        }

        // Calculate desired number of vertices per process and initialize vertex distribution
        idx_t avg_vertices = num_nodes / numProcesses;  // Integer division for base size
        idx_t remainder = num_nodes % numProcesses;     // Extra vertices to distribute
        
        // Initialize vertex distribution array
        std::vector<idx_t> vertexDistribution(numProcesses + 1, 0);
        vertexDistribution[0] = 0;
        
        // Distribute vertices more evenly, handling remainder
        for (int i = 1; i <= numProcesses; i++) {
            vertexDistribution[i] = vertexDistribution[i-1] + avg_vertices;
            if (i <= remainder) {  // Add one extra vertex to early processes if needed
                vertexDistribution[i]++;
            }
        }

        // Print the initial distribution (debug)
        if (processRank == 0) {
            std::cout << "\nInitial vertex distribution: ";
            for (int i = 0; i < numProcesses + 1; i++) {
                std::cout << vertexDistribution[i] << " ";
            }
            std::cout << std::endl;
        }

        // Broadcast adjacency information to all ranks
        // Communication pattern for distributing adjacency information:
        // 1. Root process (rank 0) broadcasts the size of the adjacency list
        // 2. Root process broadcasts the adjacency pointers array (size = num_nodes + 1)
        //    - This array contains the starting indices for each vertex's adjacency list
        // 3. Root process broadcasts the adjacency list array (size = adjListSize)
        //    - This array contains the actual adjacency information for all vertices
        // 4. Non-root processes:
        //    - First receive the size information
        //    - Allocate their local arrays based on the received size
        //    - Then receive the actual data
        if (processRank == 0) {
            idx_t adjListSize = adjacencyList.size();
            MPI_Bcast(&adjListSize, 1, MPI_INT, 0, comm);
            MPI_Bcast(adjacencyPointers.data(), num_nodes + 1, MPI_INT, 0, comm);
            MPI_Bcast(adjacencyList.data(), adjListSize, MPI_INT, 0, comm);
        } else {
            idx_t adjListSize;
            MPI_Bcast(&adjListSize, 1, MPI_INT, 0, comm);
            adjacencyPointers.resize(num_nodes + 1);
            adjacencyList.resize(adjListSize);
            MPI_Bcast(adjacencyPointers.data(), num_nodes + 1, MPI_INT, 0, comm);
            MPI_Bcast(adjacencyList.data(), adjListSize, MPI_INT, 0, comm);
        }

        // All ranks allocate vertexPartition
        // Array to store the partitioning result from ParMETIS
        // Each element vertexPartition[i] contains the process rank (0 to numProcesses-1) 
        // that vertex i is assigned to after partitioning
        // Size is num_nodes to store partition assignment for every vertex in the mesh
        std::vector<idx_t> vertexPartition(num_nodes);

        // Create vertex weights to force balanced partitioning
        std::vector<idx_t> vertexWeights(num_nodes, 1);  // All vertices have equal weight
        
        // Set options to maximum imbalance tolerance (for testing)
        std::vector<real_t> targetPartitionWeights(numProcesses, 1.0/numProcesses);
        std::vector<real_t> imbalanceTolerance(1, 1.01);  // Very strict 1% imbalance tolerance
        
        // Set ParMETIS options
        idx_t parmetisOptions[METIS_NOPTIONS];
        METIS_SetDefaultOptions(parmetisOptions);
        parmetisOptions[0] = 1;     // Use options
        parmetisOptions[1] = 0;     // No debug info
        parmetisOptions[2] = 1;     // Use Sort matching instead of Random
        parmetisOptions[3] = 1;     // Force parallel partitioning
        
        // Set weight flag to indicate we're using vertex weights
        idx_t useWeights = 1;  // Use vertex weights
        
        // Validate input before calling ParMETIS
        validate_parmetis_input(vertexDistribution, adjacencyPointers, adjacencyList, processRank);
        
        // Use METIS_PartGraphKway directly for testing on rank 0 (simpler approach)
        if (processRank == 0) {
            std::cout << "Trying direct METIS partitioning first...\n";
        
        // Initialize partition array
            std::vector<idx_t> metis_partition(num_nodes);
            
            // Call METIS directly (serial partitioner)
            idx_t n = num_nodes;
            idx_t ncon = 1;
            idx_t edgecut;
            
            int metis_result = METIS_PartGraphKway(
                &n,                      // Number of vertices
                &ncon,                   // Number of balancing constraints
                adjacencyPointers.data(),// Adjacency structure: xadj
                adjacencyList.data(),    // Adjacency structure: adjncy
                NULL,                    // Vertex weights
                NULL,                    // Size of vertices for comm volume
                NULL,                    // Edge weights
                &numProcesses,          // Number of partitions
                NULL,                    // Target partition weights
                NULL,                    // Constraints
                NULL,                    // Options
                &edgecut,                // Output: Edge-cut or comm volume
                metis_partition.data()   // Output: Partition vector
            );
            
            // Count vertices assigned to each partition by METIS
            std::vector<int> metis_partition_sizes(numProcesses, 0);
            for (idx_t i = 0; i < num_nodes; i++) {
                metis_partition_sizes[metis_partition[i]]++;
            }
            
            std::cout << "METIS partitioning result: " << (metis_result == METIS_OK ? "Success" : "Failed") << std::endl;
            std::cout << "METIS partition distribution:" << std::endl;
            for (int i = 0; i < numProcesses; i++) {
                std::cout << "Rank " << i << ": " << metis_partition_sizes[i] << " vertices" << std::endl;
            }
        }
        
        MPI_Barrier(comm);  // Make sure all ranks wait before continuing
        
        // Let's try using ParMETIS_V3_PartMeshKway which is designed for mesh partitioning
        // First create arrays for node elements - this maps nodes to elements
        std::vector<idx_t> elementsPerNode;
        std::vector<idx_t> nodeElements;
        
        if (processRank == 0) {
            // Create a mapping of nodes to elements
            elementsPerNode.resize(num_nodes, 0);
            
            // First count how many elements each node belongs to
            for (size_t e = 0; e < num_elems; e++) {
                for (size_t n = 0; n < mesh.num_nodes_in_elem; n++) {
                    idx_t node_id = mesh.nodes_in_elem.host(e, n);
                    elementsPerNode[node_id]++;
                }
            }
            
            // Calculate total size needed for nodeElements
            idx_t totalEntries = 0;
            for (size_t i = 0; i < num_nodes; i++) {
                totalEntries += elementsPerNode[i];
            }
            
            // Create nodeElements array and fill it
            nodeElements.resize(totalEntries);
            
            // Create temporary array to track current position for each node
            std::vector<idx_t> currentPos(num_nodes, 0);
            
            // Reset elementsPerNode to use as a cumulative sum
            idx_t sum = 0;
            for (size_t i = 0; i < num_nodes; i++) {
                idx_t count = elementsPerNode[i];
                elementsPerNode[i] = sum;
                sum += count;
            }
            elementsPerNode.push_back(sum);  // Add final entry
            
            // Fill in nodeElements
            for (size_t e = 0; e < num_elems; e++) {
                for (size_t n = 0; n < mesh.num_nodes_in_elem; n++) {
                    idx_t node_id = mesh.nodes_in_elem.host(e, n);
                    idx_t pos = elementsPerNode[node_id] + currentPos[node_id];
                    nodeElements[pos] = e;
                    currentPos[node_id]++;
                }
            }
        }
        
        // Broadcast node-to-element mapping to all ranks
        if (processRank == 0) {
            idx_t nePtrSize = elementsPerNode.size();
            idx_t neSize = nodeElements.size();
            MPI_Bcast(&nePtrSize, 1, MPI_INT, 0, comm);
            MPI_Bcast(&neSize, 1, MPI_INT, 0, comm);
            MPI_Bcast(elementsPerNode.data(), nePtrSize, MPI_INT, 0, comm);
            MPI_Bcast(nodeElements.data(), neSize, MPI_INT, 0, comm);
        } else {
            idx_t nePtrSize, neSize;
            MPI_Bcast(&nePtrSize, 1, MPI_INT, 0, comm);
            MPI_Bcast(&neSize, 1, MPI_INT, 0, comm);
            elementsPerNode.resize(nePtrSize);
            nodeElements.resize(neSize);
            MPI_Bcast(elementsPerNode.data(), nePtrSize, MPI_INT, 0, comm);
            MPI_Bcast(nodeElements.data(), neSize, MPI_INT, 0, comm);
        }
        
        // Now try ParMETIS_V3_PartMeshKway
        idx_t ncon = 1;
        std::vector<idx_t> elementPartition(num_elems);
        
        int result2 = ParMETIS_V3_PartMeshKway(
            elementsPerNode.data(),    // elmdist
            nodeElements.data(),       // eptr
            NULL,                      // eind (not used since we provide element to node mapping)
            NULL,                      // elmwgt
            &useWeights,               // wgtflag
            &zeroBasedIndexing,        // numflag
            &ncon,                     // ncon
            &numProcesses,            // nparts
            targetPartitionWeights.data(), // tpwgts
            imbalanceTolerance.data(), // ubvec
            parmetisOptions,           // options
            &cutEdgeCount,             // edgecut
            elementPartition.data(),   // part (output)
            &comm                      // comm
        );
        
        // Derive node partitioning from element partitioning
        if (processRank == 0) {
            // Count elements assigned to each part
            std::vector<int> elem_partition_sizes(numProcesses, 0);
            for (idx_t i = 0; i < num_elems; i++) {
                elem_partition_sizes[elementPartition[i]]++;
            }
            
            std::cout << "\nMesh partitioning result: " << (result2 == METIS_OK ? "Success" : "Failed with code " + std::to_string(result2)) << std::endl;
            std::cout << "Element partition distribution:" << std::endl;
            for (int i = 0; i < numProcesses; i++) {
                std::cout << "Rank " << i << ": " << elem_partition_sizes[i] << " elements" << std::endl;
            }
            
            // Assign each node to the partition that owns the most elements it's part of
            std::vector<idx_t> meshVertexPartition(num_nodes);
            for (idx_t i = 0; i < num_nodes; i++) {
                std::vector<int> partCount(numProcesses, 0);
                for (idx_t j = elementsPerNode[i]; j < elementsPerNode[i+1]; j++) {
                    idx_t elem = nodeElements[j];
                    partCount[elementPartition[elem]]++;
                }
                
                // Find the part with the most elements for this node
                int maxPart = 0;
                for (int p = 1; p < numProcesses; p++) {
                    if (partCount[p] > partCount[maxPart]) {
                        maxPart = p;
                    }
                }
                meshVertexPartition[i] = maxPart;
            }
            
            // Count nodes assigned to each part
            std::vector<int> mesh_partition_sizes(numProcesses, 0);
            for (idx_t i = 0; i < num_nodes; i++) {
                mesh_partition_sizes[meshVertexPartition[i]]++;
            }
            
            std::cout << "Derived node partition distribution:" << std::endl;
            for (int i = 0; i < numProcesses; i++) {
                std::cout << "Rank " << i << ": " << mesh_partition_sizes[i] << " vertices" << std::endl;
            }
        }
    }
    
    // Finalize Kokkos
    Kokkos::finalize();
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}
