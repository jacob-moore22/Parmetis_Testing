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
#include <petscdmplex.h>

static char help[] = "Minimal VTK mesh decomposition with connectivity analysis\n";


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


PetscErrorCode PrintConnectivity(DM dm)
{
    PetscInt       dim, cStart, cEnd, vStart, vEnd, fStart, fEnd;
    const PetscInt *cone;
    PetscInt       coneSize;
    PetscMPIInt    rank;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
    
    // Get point ranges for different mesh entities
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);      // vertices
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);     // cells
    if (dim > 2) {
        ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr); // faces
    }
    
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n=== RANK %d CONNECTIVITY ===\n", rank); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] Vertices: %D to %D-1 (%D total)\n", 
                      rank, vStart, vEnd, vEnd-vStart); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] Cells: %D to %D-1 (%D total)\n", 
                      rank, cStart, cEnd, cEnd-cStart); CHKERRQ(ierr);
    if (dim > 2) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] Faces: %D to %D-1 (%D total)\n", 
                          rank, fStart, fEnd, fEnd-fStart); CHKERRQ(ierr);
    }
    
    // Print cell-to-vertex connectivity (cone)
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n[%d] CELL-TO-VERTEX CONNECTIVITY (Cone):\n", rank); CHKERRQ(ierr);
    for (PetscInt c = cStart; c < cEnd; c++) {
        ierr = DMPlexGetConeSize(dm, c, &coneSize); CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, c, &cone); CHKERRQ(ierr);
        
        ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] Cell %D -> vertices: ", rank, c); CHKERRQ(ierr);
        for (PetscInt v = 0; v < coneSize; v++) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "%D ", cone[v]); CHKERRQ(ierr);
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n"); CHKERRQ(ierr);
    }
    
    // Print vertex-to-cell connectivity (support)
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n[%d] VERTEX-TO-CELL CONNECTIVITY (Support):\n", rank); CHKERRQ(ierr);
    for (PetscInt v = vStart; v < vEnd; v++) {
        const PetscInt *support;
        PetscInt supportSize;
        ierr = DMPlexGetSupportSize(dm, v, &supportSize); CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, v, &support); CHKERRQ(ierr);
        
        ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] Vertex %D -> cells: ", rank, v); CHKERRQ(ierr);
        for (PetscInt c = 0; c < supportSize; c++) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "%D ", support[c]); CHKERRQ(ierr);
        }
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n"); CHKERRQ(ierr);
    }
    
    // Print face connectivity if 3D mesh
    if (dim > 2 && fEnd > fStart) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n[%d] FACE-TO-VERTEX CONNECTIVITY:\n", rank); CHKERRQ(ierr);
        for (PetscInt f = fStart; f < PetscMin(fStart + 5, fEnd); f++) { // Show first 5 faces
            ierr = DMPlexGetConeSize(dm, f, &coneSize); CHKERRQ(ierr);
            ierr = DMPlexGetCone(dm, f, &cone); CHKERRQ(ierr);
            
            ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] Face %D -> vertices: ", rank, f); CHKERRQ(ierr);
            for (PetscInt v = 0; v < coneSize; v++) {
                ierr = PetscPrintf(PETSC_COMM_SELF, "%D ", cone[v]); CHKERRQ(ierr);
            }
            ierr = PetscPrintf(PETSC_COMM_SELF, "\n"); CHKERRQ(ierr);
        }
        if (fEnd - fStart > 5) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] ... (showing first 5 of %D faces)\n", 
                              rank, fEnd-fStart); CHKERRQ(ierr);
        }
    }
    
    // Print ghost/halo information
    PetscSF sf;
    ierr = DMGetPointSF(dm, &sf); CHKERRQ(ierr);
    if (sf) {
        PetscInt nroots, nleaves;
        const PetscInt *leaves;
        const PetscSFNode *remotes;
        
        ierr = PetscSFGetGraph(sf, &nroots, &nleaves, &leaves, &remotes); CHKERRQ(ierr);
        if (nleaves > 0) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "\n[%d] GHOST/SHARED POINTS:\n", rank); CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] %D shared points:\n", rank, nleaves); CHKERRQ(ierr);
            for (PetscInt i = 0; i < PetscMin(nleaves, 10); i++) {
                ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] Local point %D -> Rank %d, Remote point %D\n", 
                                  rank, leaves[i], remotes[i].rank, remotes[i].index); CHKERRQ(ierr);
            }
            if (nleaves > 10) {
                ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] ... (showing first 10 of %D)\n", 
                                  rank, nleaves); CHKERRQ(ierr);
            }
        }
    }
    
    ierr = PetscPrintf(PETSC_COMM_SELF, "=== END RANK %d ===\n\n", rank); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

PetscErrorCode AnalyzeCommunicationPattern(DM dm, PetscSF sf)
{
    PetscInt       nroots, nleaves;
    const PetscInt *leaves;
    const PetscSFNode *remotes;
    PetscMPIInt    rank;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
    ierr = PetscSFGetGraph(sf, &nroots, &nleaves, &leaves, &remotes); CHKERRQ(ierr);
    
    if (nleaves == 0) PetscFunctionReturn(0);
    
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n[%d] COMMUNICATION PATTERN ANALYSIS:\n", rank); CHKERRQ(ierr);
    
    // Classify ghost points by entity type
    PetscInt vStart, vEnd, cStart, cEnd, fStart, fEnd;
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);      // vertices
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);     // cells
    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd); CHKERRQ(ierr);     // faces
    
    PetscInt ghostVertices = 0, ghostCells = 0, ghostFaces = 0, ghostOther = 0;
    
    for (PetscInt i = 0; i < nleaves; i++) {
        PetscInt localPoint = leaves ? leaves[i] : i;
        
        if (localPoint >= vStart && localPoint < vEnd) {
            ghostVertices++;
        } else if (localPoint >= cStart && localPoint < cEnd) {
            ghostCells++;
        } else if (localPoint >= fStart && localPoint < fEnd) {
            ghostFaces++;
        } else {
            ghostOther++;
        }
    }
    
    ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]   Ghost vertices: %D\n", rank, ghostVertices); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]   Ghost faces: %D\n", rank, ghostFaces); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]   Ghost cells: %D\n", rank, ghostCells); CHKERRQ(ierr);
    if (ghostOther > 0) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]   Ghost other: %D\n", rank, ghostOther); CHKERRQ(ierr);
    }
    
    // Show theoretical communication pattern (without actual communication)
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n[%d] THEORETICAL COMMUNICATION PATTERN:\n", rank); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] (Showing what would be communicated without performing actual communication)\n", rank); CHKERRQ(ierr);
    
    // Show a few examples of what would be communicated
    PetscInt maxShow = PetscMin(nleaves, 5);
    for (PetscInt i = 0; i < maxShow; i++) {
        PetscInt localPoint = leaves ? leaves[i] : i;
        PetscScalar theoreticalValue = remotes[i].rank * 1000.0 + remotes[i].index;  // What would be received
        ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]   Local ghost point %D would receive value %g from rank %d\n", 
                          rank, localPoint, PetscRealPart(theoreticalValue), remotes[i].rank); CHKERRQ(ierr);
    }
    
    if (nleaves > maxShow) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]   ... (showing first %D of %D ghost points)\n", 
                          rank, maxShow, nleaves); CHKERRQ(ierr);
    }
    
    PetscFunctionReturn(0);
}


PetscErrorCode PrintCommunicationNetwork(DM dm)
{
    PetscSF        sf;
    PetscMPIInt    rank, size;
    PetscInt       nroots, nleaves;
    const PetscInt *leaves;
    const PetscSFNode *remotes;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size); CHKERRQ(ierr);
    
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n=== RANK %d COMMUNICATION NETWORK ===\n", rank); CHKERRQ(ierr);
    
    // Get the star forest from the DM
    ierr = DMGetPointSF(dm, &sf); CHKERRQ(ierr);
    
    if (!sf) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] No star forest - single process or no shared points\n", rank); CHKERRQ(ierr);
        PetscFunctionReturn(0);
    }
    
    // Get star forest graph information
    ierr = PetscSFGetGraph(sf, &nroots, &nleaves, &leaves, &remotes); CHKERRQ(ierr);
    
    ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] Star Forest Info:\n", rank); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]   Roots (owned points): %D\n", rank, nroots); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]   Leaves (ghost points): %D\n", rank, nleaves); CHKERRQ(ierr);
    
    // Analyze communication patterns
    if (nleaves > 0) {
        // Count communications per remote rank
        PetscInt *rankCounts;
        ierr = PetscCalloc1(size, &rankCounts); CHKERRQ(ierr);
        
        for (PetscInt i = 0; i < nleaves; i++) {
            rankCounts[remotes[i].rank]++;
        }
        
        ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] Communication partners:\n", rank); CHKERRQ(ierr);
        for (PetscInt r = 0; r < size; r++) {
            if (rankCounts[r] > 0) {
                ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]   -> Rank %D: %D shared points\n", 
                                  rank, r, rankCounts[r]); CHKERRQ(ierr);
            }
        }
        
        // Detailed leaf-to-root mapping
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n[%d] DETAILED STAR FOREST MAPPING:\n", rank); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] Format: Local_Ghost_Point -> (Remote_Rank, Remote_Point)\n", rank); CHKERRQ(ierr);
        
        PetscInt maxShow = PetscMin(nleaves, 15);
        for (PetscInt i = 0; i < maxShow; i++) {
            PetscInt localPoint = leaves ? leaves[i] : i;
            ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]   %D -> (%d, %D)\n", 
                              rank, localPoint, remotes[i].rank, remotes[i].index); CHKERRQ(ierr);
        }
        if (nleaves > maxShow) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]   ... (showing first %D of %D)\n", 
                              rank, maxShow, nleaves); CHKERRQ(ierr);
        }
        
        ierr = PetscFree(rankCounts); CHKERRQ(ierr);
    }
    
    // Analyze star forest type and properties
    PetscSFType sfType;
    ierr = PetscSFGetType(sf, &sfType); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] Star Forest Type: %s\n", rank, sfType); CHKERRQ(ierr);
    
    // Check if star forest is setup for communication
    PetscBool setup;
    ierr = PetscSFSetUp(sf); CHKERRQ(ierr);
    setup = PETSC_TRUE;
    ierr = PetscPrintf(PETSC_COMM_SELF, "[%d] Star Forest Setup: %s\n", rank, setup ? "Yes" : "No"); CHKERRQ(ierr);
    
    // Demonstrate communication pattern analysis
    ierr = AnalyzeCommunicationPattern(dm, sf); CHKERRQ(ierr);
    
    ierr = PetscPrintf(PETSC_COMM_SELF, "=== END RANK %d COMMUNICATION ===\n\n", rank); CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}



int main(int argc, char *argv[]) {
    // Initialize MPI - required for distributed memory parallelism
    MPI_Init(&argc, &argv);
    
    // Initialize Kokkos - required for shared memory parallelism
    Kokkos::initialize(argc, argv);
    {

        DM             dm, dmDist;
        PetscViewer    viewer;
        char           infile[PETSC_MAX_PATH_LEN] = "../bird.msh";
        char           outfile[PETSC_MAX_PATH_LEN] = "output.vtu";
        PetscMPIInt    rank;
        PetscErrorCode ierr;

        // Initialize PETSc
        ierr = PetscInitialize(&argc, &argv, nullptr, help); if (ierr) return ierr;
        ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

        // Get input/output filenames from command line
        ierr = PetscOptionsGetString(nullptr, nullptr, "-i", infile, sizeof(infile), nullptr); CHKERRQ(ierr);
        ierr = PetscOptionsGetString(nullptr, nullptr, "-o", outfile, sizeof(outfile), nullptr); CHKERRQ(ierr);

        // Read VTK mesh file
        ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, infile, nullptr, PETSC_TRUE, &dm); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Loaded mesh from %s\n", infile); CHKERRQ(ierr);

        // Decompose mesh across processes
        ierr = DMPlexDistribute(dm, 0, nullptr, &dmDist); CHKERRQ(ierr);
        if (dmDist) {
            ierr = DMDestroy(&dm); CHKERRQ(ierr);
            dm = dmDist;
            ierr = PetscPrintf(PETSC_COMM_WORLD, "Mesh distributed\n"); CHKERRQ(ierr);
        }

        // Print connectivity information per rank
        // ierr = PrintConnectivity(dm); CHKERRQ(ierr);
        
        // Print communication network and star forest details
        ierr = PrintCommunicationNetwork(dm); CHKERRQ(ierr);

        // Export decomposed mesh
        ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer); CHKERRQ(ierr);
        ierr = PetscViewerSetType(viewer, PETSCVIEWERVTK); CHKERRQ(ierr);
        ierr = PetscViewerFileSetMode(viewer, FILE_MODE_WRITE); CHKERRQ(ierr);
        ierr = PetscViewerFileSetName(viewer, outfile); CHKERRQ(ierr);
        ierr = DMView(dm, viewer); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Exported decomposed mesh to %s\n", outfile); CHKERRQ(ierr);

        // Cleanup
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
        ierr = DMDestroy(&dm); CHKERRQ(ierr);
        ierr = PetscFinalize();
        return ierr;

    } // end Kokkos initialize
    // Finalize Kokkos
    Kokkos::finalize();
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}

/*
 * Compile:
 * mpicxx -o decompose decompose.cpp -I$PETSC_DIR/include -L$PETSC_DIR/lib -lpetsc
 * 
 * Run:
 * mpirun -np 4 ./petsc_test -i mesh.vtk -o decomposed.vtk
 *  
 * PETSC CONNECTIVITY STRUCTURE EXPLANATION:
 * =========================================
 * 
 * PETSc DMPlex uses a "point" numbering system where all mesh entities 
 * (vertices, edges, faces, cells) are numbered as "points" in ranges:
 * 
 * 1. POINT NUMBERING:
 *    - Vertices: depth 0 (bottom of topology)
 *    - Edges: depth 1 (1D entities) 
 *    - Faces: depth 2 (2D entities)
 *    - Cells: depth 3 (3D entities, height 0 from top)
 * 
 * 2. CONE/SUPPORT RELATIONSHIPS:
 *    - Cone(p): Points that bound point p "from below"
 *      * Cone(cell) = its faces
 *      * Cone(face) = its edges  
 *      * Cone(edge) = its vertices
 *    - Support(p): Points that are bounded by point p "from above"
 *      * Support(vertex) = edges containing it
 *      * Support(edge) = faces containing it
 *      * Support(face) = cells containing it
 * 
 * 3. HEXAHEDRAL EXAMPLE:
 *    For a hex cell with ID 100:
 *    - Cone(100) = [face_1, face_2, face_3, face_4, face_5, face_6]
 *    - Each face has Cone(face_i) = [edge_a, edge_b, edge_c, edge_d]  
 *    - Each edge has Cone(edge_j) = [vertex_x, vertex_y]
 * 
 * 4. PARALLEL DECOMPOSITION:
 *    - Each rank owns a subset of cells and their closure (vertices/faces)
 *    - Shared entities (vertices/faces on partition boundaries) exist on multiple ranks
 *    - PetscSF (Star Forest) manages communication between shared entities
 *    - Ghost points: local numbering for entities owned by other ranks
 * 
 * 5. POINT RANGES BY RANK:
 *    - Point numbers are LOCAL to each rank
 *    - Same global entity may have different local numbers on different ranks
 *    - DMPlexGetDepthStratum/HeightStratum gives local ranges per entity type
 * 
 * 6. INTERPOLATION:
 *    - Creates intermediate entities (faces, edges) if not present
 *    - Required for most finite element computations
 *    - Enables neighbor-finding and boundary detection
 * 
 * STAR FOREST (PetscSF) COMMUNICATION DETAILS:
 * ===========================================
 * 
 * The Star Forest is PETSc's fundamental communication abstraction for
 * distributed data structures. It manages all inter-rank communication
 * for shared mesh entities.
 * 
 * 1. STAR FOREST CONCEPT:
 *    - "Roots": Points owned by this rank (can send data to others)
 *    - "Leaves": Ghost/halo points owned by remote ranks (receive data)
 *    - Forms a star-shaped communication pattern: one root -> many leaves
 * 
 * 2. COMMUNICATION OPERATIONS:
 *    - Broadcast (BcastBegin/End): Root -> Leaves (scatter operation)
 *      * Send data from owned points to their ghost copies on other ranks
 *      * Used for: field updates, coordinate sharing, boundary conditions
 *    - Reduce (ReduceBegin/End): Leaves -> Root (gather operation)
 *      * Accumulate contributions from ghost points back to owner
 *      * Used for: assembly operations, residual computation, mass conservation
 * 
 * 3. PARALLEL MESH COMMUNICATION PATTERNS:
 *    Rank 0: owns cells [0,1], needs ghosts from rank 1
 *    Rank 1: owns cells [2,3], needs ghosts from rank 0
 *    
 *    Communication graph:
 *    Rank 0 ←→ Rank 1  (bidirectional ghost exchange)
 *    
 *    Star Forest on Rank 0:
 *    - Roots: local points [0...N] that rank 0 owns
 *    - Leaves: ghost points that are owned by rank 1
 *    - Mapping: local_ghost_point -> (remote_rank, remote_point)
 * 
 * 4. MESH PARTITION BOUNDARIES:
 *    - Vertices on partition boundaries are shared (appear on multiple ranks)
 *    - Faces between partitions exist on both adjacent ranks
 *    - One rank is designated "owner" for each shared entity
 *    - Non-owners have "ghost" copies that receive updates from owner
 * 
 * 5. COMMUNICATION EFFICIENCY:
 *    - PETSc optimizes communication by batching all exchanges
 *    - Uses non-blocking MPI operations when possible
 *    - Minimizes number of messages (one per communicating rank pair)
 *    - Supports different SF implementations (Basic, Neighbor, Window)
 * 
 * 6. TYPICAL USAGE IN FINITE ELEMENT/VOLUME METHODS:
 *    a) Assembly phase: 
 *       - Compute local contributions to global system
 *       - Use SF Reduce to sum contributions from ghosts to owners
 *    b) Solution update phase:
 *       - Update owned DOF values from linear solver
 *       - Use SF Broadcast to send updates to ghost copies
 *    c) Residual computation:
 *       - Need current values at ghost points for gradient calculation
 *       - SF Broadcast ensures ghosts have current data
 * 
 * 7. STAR FOREST TYPES IN PETSC:
 *    - PETSCSFBASIC: Standard implementation using MPI point-to-point
 *    - PETSCSFNEIGHBOR: Uses MPI neighborhood collectives (MPI-3)
 *    - PETSCSFWINDOW: Uses MPI one-sided operations (MPI-2)
 *    - Choice affects performance but not functionality
 */