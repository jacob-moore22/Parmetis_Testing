import meshio

# Read the mesh_Sedov_8.geo file
mesh = meshio.read("bird.vtk")

# Write it as mesh_Sedov_8.msh
mesh.write("bird.msh",file_format="gmsh22")  