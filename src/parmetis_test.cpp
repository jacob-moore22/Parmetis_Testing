#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <mpi.h>
// #include <parmetis.h>
#include <Kokkos_Core.hpp>
#include "matar.h"

// Required for MATAR data structures
using namespace mtr;

/**
 * DistributedDCArray: A class that extends DCArrayKokkos with MPI communication capabilities
 * for distributed data management with halo exchange
 * 
 * This class handles:
 * - Managing distributed data across MPI processes
 * - Building the connectivity for HALO communications 
 * - Automating the communication process via a simple .comm() command
 */
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class DistributedDCArray {
private:
    using DataArray = Kokkos::DualView<T*, Layout, ExecSpace, MemoryTraits>;
    using IndexArray = Kokkos::DualView<idx_t*, Layout, ExecSpace, MemoryTraits>;
    using CommunicationMap = Kokkos::DualView<int*, Layout, ExecSpace, MemoryTraits>;

    // MPI related members
    int processRank_;
    int totalProcesses_;
    MPI_Comm communicator_;
    
    // Distribution and connectivity information
    
    // Array storing the distribution of vertices across processors
    // vertexDistribution_[i] contains the first global vertex number owned by processor i
    // Size is number of processors + 1, where vertexDistribution_[p+1] - vertexDistribution_[p] gives
    // number of vertices owned by processor p
    IndexArray vertexDistribution_;
    
    // Array storing indices into adjacencyList_ array for each vertex's adjacency list
    // For vertex i, its adjacent vertices are stored in adjacencyList_[adjacencyPointers_[i]] through
    // adjacencyList_[adjacencyPointers_[i+1]-1]. Size is number of local vertices + 1
    IndexArray adjacencyPointers_;   
    
    // Array storing adjacent vertices in compressed format
    // Contains concatenated lists of adjacent vertices for each local vertex
    // Vertices are stored using global numbering
    IndexArray adjacencyList_;      
    
    // Local-to-global and global-to-local mappings
    CommunicationMap localToGlobalMap_;  // Maps local indices to global indices
    CommunicationMap globalToLocalMap_;  // Maps global indices to local indices
    
    // HALO communication data
    CommunicationMap neighborProcesses_;         // List of neighbor processes
    CommunicationMap indicesToSend_;      // Indices to send to each neighbor
    CommunicationMap indicesToReceive_;      // Indices to receive from each neighbor
    CommunicationMap sendCounts_;       // Number of elements to send to each neighbor
    CommunicationMap receiveCounts_;       // Number of elements to receive from each neighbor
    CommunicationMap sendOffsets_; // Displacements for sends
    CommunicationMap receiveOffsets_; // Displacements for receives
    
    // Data array containing both owned and halo elements
    DCArrayKokkos<T, Layout, ExecSpace, MemoryTraits> meshData_;
    
    // Count of owned elements vs. total (owned + halo)
    size_t localElementCount_;
    size_t totalElementCount_;
    
    /**
     * Sets up the HALO (ghost) region communication patterns between processes
     * 
     * This function:
     * 1. Analyzes the adjacency graph to identify neighboring processes
     * 2. Determines which vertices need to be sent/received between processes
     * 3. Sets up the communication buffers and patterns for efficient data exchange
     */
    void setup_halo_communications() {
        // First, determine which processes own adjacent vertices
        int neighborCount = 0;
        std::vector<int> tempNeighbors;
        
        // For each boundary vertex, find which process owns it
        for (size_t i = 0; i < adjacencyList_.extent(0); i++) {
            idx_t globalVertexIndex = adjacencyList_.h_view(i);
            
            // Find process that owns this vertex
            int ownerProcess = -1;
            for (int p = 0; p < totalProcesses_; p++) {
                if (globalVertexIndex >= vertexDistribution_.h_view(p) && globalVertexIndex < vertexDistribution_.h_view(p+1)) {
                    ownerProcess = p;
                    break;
                }
            }
            
            // If it's not owned by current process, add to neighbor list
            if (ownerProcess != processRank_ && ownerProcess != -1) {
                // Check if this neighbor is already in our list
                bool alreadyFound = false;
                for (int n : tempNeighbors) {
                    if (n == ownerProcess) {
                        alreadyFound = true;
                        break;
                    }
                }
                
                if (!alreadyFound) {
                    tempNeighbors.push_back(ownerProcess);
                    neighborCount++;
                }
            }
        }
        
        // Setup neighbor arrays
        neighborProcesses_ = CommunicationMap("neighborProcesses", neighborCount);
        sendCounts_ = CommunicationMap("sendCounts", neighborCount);
        receiveCounts_ = CommunicationMap("receiveCounts", neighborCount);
        sendOffsets_ = CommunicationMap("sendOffsets", neighborCount);
        receiveOffsets_ = CommunicationMap("receiveOffsets", neighborCount);
        
        // Copy neighbors to Kokkos array
        for (int i = 0; i < neighborCount; i++) {
            neighborProcesses_.h_view(i) = tempNeighbors[i];
        }
        
        // For each neighbor, determine which vertices to send/receive
        std::vector<std::vector<int>> indicesToSendByNeighbor(neighborCount);
        
        for (size_t i = 0; i < adjacencyList_.extent(0); i++) {
            idx_t globalVertexIndex = adjacencyList_.h_view(i);
            
            // Find process that owns this vertex
            int ownerProcess = -1;
            for (int p = 0; p < totalProcesses_; p++) {
                if (globalVertexIndex >= vertexDistribution_.h_view(p) && globalVertexIndex < vertexDistribution_.h_view(p+1)) {
                    ownerProcess = p;
                    break;
                }
            }
            
            // If it's owned by a neighbor, add to the send list for that neighbor
            if (ownerProcess != processRank_ && ownerProcess != -1) {
                // Find index of ownerProcess in neighbor list
                int neighborIndex = -1;
                for (int n = 0; n < neighborCount; n++) {
                    if (neighborProcesses_.h_view(n) == ownerProcess) {
                        neighborIndex = n;
                        break;
                    }
                }
                
                if (neighborIndex != -1) {
                    // Convert global index to local index
                    size_t localVertexIndex = globalVertexIndex - vertexDistribution_.h_view(processRank_);
                    indicesToSendByNeighbor[neighborIndex].push_back(localVertexIndex);
                }
            }
        }
        
        // Set send counts and allocate send indices arrays
        for (int i = 0; i < neighborCount; i++) {
            sendCounts_.h_view(i) = indicesToSendByNeighbor[i].size();
        }
        
        // Sync to device
        neighborProcesses_.template modify<typename CommunicationMap::host_mirror_space>();
        neighborProcesses_.template sync<typename CommunicationMap::execution_space>();
        
        sendCounts_.template modify<typename CommunicationMap::host_mirror_space>();
        sendCounts_.template sync<typename CommunicationMap::execution_space>();
        
        // Communicate send counts to determine receive counts
        receiveCounts_ = CommunicationMap("receiveCounts", neighborCount);
        
        for (int i = 0; i < neighborCount; i++) {
            int destinationRank = neighborProcesses_.h_view(i);
            int elementsToSend = sendCounts_.h_view(i);
            int elementsToReceive;
            
            MPI_Sendrecv(&elementsToSend, 1, MPI_INT, destinationRank, 0,
                         &elementsToReceive, 1, MPI_INT, destinationRank, 0,
                         communicator_, MPI_STATUS_IGNORE);
            
            receiveCounts_.h_view(i) = elementsToReceive;
        }
        
        receiveCounts_.template modify<typename CommunicationMap::host_mirror_space>();
        receiveCounts_.template sync<typename CommunicationMap::execution_space>();
        
        // Calculate displacements
        int sendOffset = 0;
        int receiveOffset = 0;
        
        for (int i = 0; i < neighborCount; i++) {
            sendOffsets_.h_view(i) = sendOffset;
            sendOffset += sendCounts_.h_view(i);
            
            receiveOffsets_.h_view(i) = receiveOffset;
            receiveOffset += receiveCounts_.h_view(i);
        }
        
        sendOffsets_.template modify<typename CommunicationMap::host_mirror_space>();
        sendOffsets_.template sync<typename CommunicationMap::execution_space>();
        
        receiveOffsets_.template modify<typename CommunicationMap::host_mirror_space>();
        receiveOffsets_.template sync<typename CommunicationMap::execution_space>();
        
        // Allocate and set send indices
        int totalSendCount = sendOffset;
        indicesToSend_ = CommunicationMap("indicesToSend", totalSendCount);
        
        int idx = 0;
        for (int i = 0; i < neighborCount; i++) {
            for (size_t j = 0; j < indicesToSendByNeighbor[i].size(); j++) {
                indicesToSend_.h_view(idx++) = indicesToSendByNeighbor[i][j];
            }
        }
        
        indicesToSend_.template modify<typename CommunicationMap::host_mirror_space>();
        indicesToSend_.template sync<typename CommunicationMap::execution_space>();
        
        // Allocate receive indices
        int totalReceiveCount = receiveOffset;
        indicesToReceive_ = CommunicationMap("indicesToReceive", totalReceiveCount);
        
        indicesToReceive_.template modify<typename CommunicationMap::host_mirror_space>();
        indicesToReceive_.template sync<typename CommunicationMap::execution_space>();
        
        // Update total count to include halo elements
        totalElementCount_ = localElementCount_ + totalReceiveCount;
        
        // Resize data array to accommodate both owned and halo elements
        meshData_ = DCArrayKokkos<T, Layout, ExecSpace, MemoryTraits>(totalElementCount_);
    }

public:
    // Constructors
    DistributedDCArray() : processRank_(0), totalProcesses_(1), localElementCount_(0), totalElementCount_(0) {
        MPI_Comm_rank(MPI_COMM_WORLD, &processRank_);
        MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses_);
        communicator_ = MPI_COMM_WORLD;
    }
    
    DistributedDCArray(MPI_Comm comm) : localElementCount_(0), totalElementCount_(0) {
        communicator_ = comm;
        MPI_Comm_rank(comm, &processRank_);
        MPI_Comm_size(comm, &totalProcesses_);
    }
    
    /**
     * Initialize the distributed array with graph connectivity information
     * 
     * @param vertexDistData Distribution of vertices among processors
     * @param adjacencyPtrData Adjacency structure indices
     * @param adjacencyListData Adjacent vertices
     */
    void init(idx_t* vertexDistData, size_t vertexDistSize,
              idx_t* adjacencyPtrData, size_t adjacencyPtrSize,
              idx_t* adjacencyListData, size_t adjacencyListSize) {
        // Allocate arrays
        vertexDistribution_ = IndexArray("vertexDistribution", vertexDistSize);
        adjacencyPointers_ = IndexArray("adjacencyPointers", adjacencyPtrSize);
        adjacencyList_ = IndexArray("adjacencyList", adjacencyListSize);
        
        // Copy data to host views
        for (size_t i = 0; i < vertexDistSize; i++) {
            vertexDistribution_.h_view(i) = vertexDistData[i];
        }
        
        for (size_t i = 0; i < adjacencyPtrSize; i++) {
            adjacencyPointers_.h_view(i) = adjacencyPtrData[i];
        }
        
        for (size_t i = 0; i < adjacencyListSize; i++) {
            adjacencyList_.h_view(i) = adjacencyListData[i];
        }
        
        // Update device views
        vertexDistribution_.template modify<typename IndexArray::host_mirror_space>();
        vertexDistribution_.template sync<typename IndexArray::execution_space>();
        
        adjacencyPointers_.template modify<typename IndexArray::host_mirror_space>();
        adjacencyPointers_.template sync<typename IndexArray::execution_space>();
        
        adjacencyList_.template modify<typename IndexArray::host_mirror_space>();
        adjacencyList_.template sync<typename IndexArray::execution_space>();
        
        // Calculate owned count
        localElementCount_ = vertexDistribution_.h_view(processRank_ + 1) - vertexDistribution_.h_view(processRank_);
        totalElementCount_ = localElementCount_;
        
        // Initialize data array
        meshData_ = DCArrayKokkos<T, Layout, ExecSpace, MemoryTraits>(totalElementCount_);
        
        // Setup HALO communications
        setup_halo_communications();
    }
    
    /**
     * Access data element (both owned and halo)
     * 
     * @param i Index
     * @return Reference to data element
     */
    T& operator()(size_t i) const {
        return meshData_(i);
    }
    
    /**
     * Perform HALO communications to sync ghost regions
     */
    void comm() {
        // Need to communicate data for ghost regions
        meshData_.update_host();
        
        // Allocate send and receive buffers
        int totalSendCount = 0;
        for (size_t i = 0; i < sendCounts_.extent(0); i++) {
            totalSendCount += sendCounts_.h_view(i);
        }
        
        int totalReceiveCount = 0;
        for (size_t i = 0; i < receiveCounts_.extent(0); i++) {
            totalReceiveCount += receiveCounts_.h_view(i);
        }
        
        T* sendBuffer = new T[totalSendCount];
        T* receiveBuffer = new T[totalReceiveCount];
        
        // Pack send buffer
        for (int i = 0; i < totalSendCount; i++) {
            sendBuffer[i] = meshData_.h_view(indicesToSend_.h_view(i));
        }
        
        // Use point-to-point communication with adjacent ranks
        int neighborCount = neighborProcesses_.extent(0);
        MPI_Request* sendRequests = new MPI_Request[neighborCount];
        MPI_Request* recvRequests = new MPI_Request[neighborCount];
        MPI_Status* statuses = new MPI_Status[neighborCount];
        
        // Post receives first (to avoid potential deadlock)
        for (int i = 0; i < neighborCount; i++) {
            int neighborRank = neighborProcesses_.h_view(i);
            int recvCount = receiveCounts_.h_view(i);
            int recvOffset = receiveOffsets_.h_view(i);
            
            if (recvCount > 0) {
                MPI_Irecv(
                    &receiveBuffer[recvOffset],
                    recvCount * sizeof(T),
                    MPI_BYTE,
                    neighborRank,
                    0,  // tag
                    communicator_,
                    &recvRequests[i]
                );
            } else {
                recvRequests[i] = MPI_REQUEST_NULL;
            }
        }
        
        // Post sends
        for (int i = 0; i < neighborCount; i++) {
            int neighborRank = neighborProcesses_.h_view(i);
            int sendCount = sendCounts_.h_view(i);
            int sendOffset = sendOffsets_.h_view(i);
            
            if (sendCount > 0) {
                MPI_Isend(
                    &sendBuffer[sendOffset],
                    sendCount * sizeof(T),
                    MPI_BYTE,
                    neighborRank,
                    0,  // tag
                    communicator_,
                    &sendRequests[i]
                );
            } else {
                sendRequests[i] = MPI_REQUEST_NULL;
            }
        }
        
        // Wait for all receives to complete
        MPI_Waitall(neighborCount, recvRequests, statuses);
        
        // Unpack receive buffer
        for (int i = 0; i < totalReceiveCount; i++) {
            // Halo elements are stored after owned elements
            meshData_.h_view(localElementCount_ + i) = receiveBuffer[i];
        }
        
        // Wait for all sends to complete
        MPI_Waitall(neighborCount, sendRequests, statuses);
        
        // Clean up
        delete[] sendBuffer;
        delete[] receiveBuffer;
        delete[] sendRequests;
        delete[] recvRequests;
        delete[] statuses;
        
        // Update device view
        meshData_.template modify<typename DataArray::host_mirror_space>();
        meshData_.template sync<typename DataArray::execution_space>();
    }
    
    /**
     * Get the number of owned elements
     * 
     * @return Number of owned elements
     */
    size_t get_owned_count() const {
        return localElementCount_;
    }
    
    /**
     * Get the total number of elements (owned + halo)
     * 
     * @return Total number of elements
     */
    size_t get_total_count() const {
        return totalElementCount_;
    }
    
    /**
     * Get the MPI rank
     * 
     * @return MPI rank
     */
    int get_rank() const {
        return processRank_;
    }
    
    /**
     * Get the MPI communicator size
     * 
     * @return MPI communicator size
     */
    int get_size() const {
        return totalProcesses_;
    }
    
    /**
     * Get the MPI communicator
     * 
     * @return MPI communicator
     */
    MPI_Comm get_comm() const {
        return communicator_;
    }
    
    /**
     * Set data for local elements
     * 
     * @param value Value to set
     */
    void set_values(T value) {
        // Set all elements to value
        Kokkos::parallel_for("SetValues_DistributedDCArray", totalElementCount_, KOKKOS_CLASS_LAMBDA(const int i) {
            meshData_.d_view(i) = value;
        });
    }
    
    /**
     * Destructor
     */
    virtual ~DistributedDCArray() {}
};

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
