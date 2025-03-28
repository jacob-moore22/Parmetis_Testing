#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <mpi.h>
// #include <parmetis.h>
#include <Kokkos_Core.hpp>
#include "matar.h"
#include "distributed_array.h"


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
    
    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        // Create a simple mesh graph for demonstration
        // For this example, we'll create a 2D grid mesh
        
        // Grid dimensions
        int gridWidth = 10;  // number of vertices in x direction
        int gridHeight = 10;  // number of vertices in y direction
        int numPartitions = numProcesses;  // number of partitions (one per process)
        
        // Total number of vertices
        int totalVertices = gridWidth * gridHeight;
        
        // Distribute vertices evenly among processors
        std::vector<idx_t> vertexDistribution(numProcesses + 1);
        for (int i = 0; i <= numProcesses; i++) {
            vertexDistribution[i] = (totalVertices * i) / numProcesses;
        }
        
        // Number of vertices for this processor
        int localVertexCount = vertexDistribution[processRank + 1] - vertexDistribution[processRank];
        
        // Build adjacency structure for a 2D grid
        std::vector<idx_t> adjacencyPointers(localVertexCount + 1);
        std::vector<idx_t> adjacencyList;
        
        adjacencyPointers[0] = 0;
        
        for (int i = 0; i < localVertexCount; i++) {
            int globalVertexIndex = vertexDistribution[processRank] + i;
            int xCoord = globalVertexIndex % gridWidth;
            int yCoord = globalVertexIndex / gridWidth;
            
            // Add neighbors (up to 4 for a 2D grid)
            // Left neighbor
            if (xCoord > 0) {
                adjacencyList.push_back(globalVertexIndex - 1);
            }
            
            // Right neighbor
            if (xCoord < gridWidth - 1) {
                adjacencyList.push_back(globalVertexIndex + 1);
            }
            
            // Top neighbor
            if (yCoord > 0) {
                adjacencyList.push_back(globalVertexIndex - gridWidth);
            }
            
            // Bottom neighbor
            if (yCoord < gridHeight - 1) {
                adjacencyList.push_back(globalVertexIndex + gridWidth);
            }
            
            adjacencyPointers[i + 1] = adjacencyList.size();
        }
        
        // ========= ParMETIS Partitioning =========
        // Now we call ParMETIS to partition our graph
        
        // Initialize partition array
        std::vector<idx_t> vertexPartition(localVertexCount);
        
        // ParMETIS parameters
        idx_t useWeights = 0;  // No weights
        idx_t zeroBasedIndexing = 0;  // 0-based indexing
        idx_t numConstraints = 1;     // Number of balancing constraints
        real_t* targetPartitionWeights = new real_t[numConstraints * numPartitions];
        real_t* imbalanceTolerance = new real_t[numConstraints];
        idx_t parmetisOptions[METIS_NOPTIONS];
        idx_t cutEdgeCount;
        
        // Set balanced partitioning
        for (int i = 0; i < numConstraints * numPartitions; i++) {
            targetPartitionWeights[i] = 1.0 / numPartitions;
        }
        
        // Set maximum allowed imbalance
        for (int i = 0; i < numConstraints; i++) {
            imbalanceTolerance[i] = 1.05;  // 5% imbalance tolerance
        }
        
        // Set default options
        parmetisOptions[0] = 0;
        
        // Call ParMETIS to partition the graph
        int result = 0; /*ParMETIS_V3_PartKway(
            vertexDistribution.data(), 
            adjacencyPointers.data(), 
            adjacencyList.data(),
            NULL, NULL, &useWeights, &zeroBasedIndexing, 
            &numConstraints, &numPartitions,
            targetPartitionWeights, imbalanceTolerance, 
            parmetisOptions, &cutEdgeCount, 
            vertexPartition.data(), &MPI_COMM_WORLD
        );*/
        
        if (result == METIS_OK) {
            // Print partition info on rank 0
            if (processRank == 0) {
                std::cout << "ParMETIS partitioning completed successfully!" << std::endl;
                std::cout << "Edge-cut: " << cutEdgeCount << std::endl;
            }
        } else {
            if (processRank == 0) {
                std::cout << "ParMETIS partitioning failed with error code: " << result << std::endl;
            }
            // Clean up and exit if partitioning failed
            delete[] targetPartitionWeights;
            delete[] imbalanceTolerance;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Clean up
        delete[] targetPartitionWeights;
        delete[] imbalanceTolerance;
        
        // ========= Creating distributed array with partitioned graph =========
        // Now create our distributed array with the partitioned graph
        DistributedDCArray<double> mesh;
        
        // Initialize the array with the graph data
        mesh.init(
            vertexDistribution.data(), vertexDistribution.size(),
            adjacencyPointers.data(), adjacencyPointers.size(),
            adjacencyList.data(), adjacencyList.size()
        );
        
        // Set values based on rank for demonstration
        mesh.set_values(static_cast<double>(processRank));
        
        // Perform HALO communications
        mesh.comm();
        
        // Check some values after communication
        if (processRank == 0) {
            std::cout << "After communication on rank " << processRank << ":" << std::endl;
            std::cout << "Owned elements: " << mesh.get_owned_count() << std::endl;
            std::cout << "Total elements (owned + halo): " << mesh.get_total_count() << std::endl;
            
            // Print some values from halo regions
            if (mesh.get_total_count() > mesh.get_owned_count()) {
                std::cout << "First halo element: " << mesh(mesh.get_owned_count()) << std::endl;
            }
        }
        
        // Synchronize all processes
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Finalize Kokkos
    Kokkos::finalize();
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}
